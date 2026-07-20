# DreamZero 项目与 Baseline 入门

本文面向第一次阅读 DreamZero 代码的开发者，目标是解释：

- DreamZero 到底在预测什么；
- autoregressive 发生在哪个维度；
- video frame、latent frame、block 和 action chunk 分别是什么；
- causal KV cache 与 DiT cache 有什么区别；
- 当前 DreamZero-DROID checkpoint 的真实默认配置是什么；
- 仓库中的 demo 和后续实验 baseline 分别代表什么。

本文以当前仓库的 PyTorch 推理路径和本地 `DreamZero-DROID` checkpoint 为准。

## 1. 一句话理解 DreamZero

DreamZero 是一个 **chunk 级 autoregressive 的 World Action Model（WAM）**：

> 模型根据语言、相机观测、当前机器人状态和有限的视频历史，在一次 diffusion 推理中联合生成未来视频和一段 24-step 动作；机器人获得新观测后，再调用模型生成下一个动作 chunk。

这里最重要的是：

- autoregressive 发生在连续的 **chunk/策略调用** 之间；
- 一个 chunk 内的 24 个动作由 diffusion **一起去噪生成**；
- 它不是像 LLM 那样依次生成 action 0、action 1、action 2。

## 2. 为什么叫 World Action Model

普通 VLA 通常直接完成：

```text
图像 + 语言 + 状态 -> 动作
```

DreamZero 同时完成：

```text
图像 + 语言 + 状态 -> 未来视频 + 动作
```

模型共享一个 Causal Wan DiT，同时预测：

- `video flow`：未来视频 latent 应该如何从噪声变化为合理的视频；
- `action flow`：动作 latent 应该如何从噪声变化为合理的动作。

训练时同时优化 dynamics loss 和 action loss。未来视频分支迫使模型学习物体运动、接触后变化和场景动力学，动作分支则将这种表示转换成机器人控制目标。

## 3. 模型总体结构

```text
语言指令
   │
   ▼
UMT5 Text Encoder ─────────────────────────┐
                                           │
三路相机 ──> DROID 2×2 拼图 ──> CLIP ─────┤
                     │                     │
                     └──────> Video VAE ───┤
                                           │
当前关节/夹爪 ──> State Encoder ───────────┤
                                           │
随机未来视频噪声 + 随机动作噪声 ───────────┤
                                           ▼
                                    Causal Wan DiT
                                           │
                         ┌─────────────────┴─────────────────┐
                         ▼                                   ▼
                    video flow                          action flow
                         │                                   │
                    视频 scheduler                      动作 scheduler
                         │                                   │
                         ▼                                   ▼
                    未来视频 latent                   24-step 动作 chunk
```

当前 DROID checkpoint 的主要组成是：

- Wan2.1 I2V 风格的 40 层 Causal Wan DiT；
- UMT5-XXL 文本编码器；
- CLIP 图像编码器；
- Wan Video VAE；
- state encoder、action encoder 和 action decoder；
- 外部使用 7 维关节位置和 1 维夹爪动作。

checkpoint 的 backbone 是 `IdentityBackbone`，主要计算都在 action head 内完成。

## 4. 三路相机如何进入模型

DROID 使用：

- exterior camera 1；
- exterior camera 2；
- wrist camera。

模型不会分别把三路视频送进三个独立 DiT，而是先拼成一张 2x2 画面：

```text
┌──────────────────────────┐
│       wrist camera       │
├─────────────┬────────────┤
│ exterior 1  │ exterior 2 │
└─────────────┴────────────┘
```

wrist 图像被横向扩展，占据整行；两个外部相机位于下方。实现位于：

```text
groot/vla/model/dreamzero/transform/dreamzero_cotrain.py::_prepare_video
```

当前服务收到的每路图像是 `180x320`，经过数据变换后，模型日志中常见的拼图尺寸为 `352x640`。

## 5. Token、Frame、Block 和 Chunk

这几个词是项目中最容易混淆的部分。

### 5.1 Video token

图像经过 VAE 后成为低分辨率 latent，再经过 DiT patch embedding 切成 token。当前 checkpoint 的 `frame_seqlen=880`，即一个 latent frame 对应 880 个空间 token。

### 5.2 Video block

当前 checkpoint 配置：

```text
num_frame_per_block = 2
```

即 2 个连续 latent frame 组成一个 video block。

### 5.3 Action chunk

当前 checkpoint 配置：

```text
num_action_per_block = 24
action_horizon = 24
```

一个 video block 对应一个 24-step action chunk。

### 5.4 一个 Chunk 的数字对应关系

对于当前 DROID 数据和 checkpoint，可以这样记：

```text
1 action chunk
= 24 个控制动作
= 训练时 8 个采样视频帧
= VAE 时间压缩后的 2 个 latent frame
```

训练数据每个 chunk 从 24 个控制时刻中采样 8 帧：

```text
[0, 3, 6, 9, 12, 15, 18, 21]
```

相关采样逻辑位于：

```text
groot/vla/data/dataset/lerobot_sharded.py::_uniform_sample_from_language_ranges
```

## 6. 训练时的多 Chunk 窗口

本地 checkpoint 的真实配置为：

```text
max_chunk_size = 4
num_frames = 33
num_frame_per_block = 2
num_action_per_block = 24
```

训练视频窗口为：

```text
1 个边界/初始 RGB frame + 4 个 chunk × 8 个 RGB frame = 33 个 RGB frame
```

经过 VAE 时间压缩后变为：

```text
1 个初始 latent frame + 4 个 block × 2 latent frame = 9 个 latent frame
```

对应动作最多为：

```text
4 个 chunk × 24 个动作 = 96 个训练动作位置
```

训练使用 blockwise causal mask，因此较晚 block 可以利用较早 video block 的上下文，但不能看到未来 block。

## 7. 推理时为何只输出一个 Chunk

虽然训练样本可以包含最多 4 个 chunk，在线推理一次只生成当前的一个 action chunk：

```text
当前观测 + 历史 KV + 当前状态
    -> 预测 2 个未来 latent frame
    -> 预测 24 个未来动作
```

机器人执行动作并取得新观测后，再调用一次模型：

```text
调用 0: observation 0 + state 0 -> action chunk 0
执行动作并获得新观测

调用 1: observation 1 + state 1 + cached history -> action chunk 1
执行动作并获得新观测

调用 2: observation 2 + state 2 + cached history -> action chunk 2
...
```

因此 DreamZero 是 **chunk-level autoregressive**，不是 action-token-level autoregressive。

## 8. DROID Demo 的帧输入

`test_client_AR.py` 的第一步发送单帧 `frame 0`，用于初始化 session、CLIP 条件和 causal KV cache。

后续每个请求发送 4 个稀疏历史帧：

```text
chunk 0: [0, 7, 15, 23]
chunk 1: [24, 31, 39, 47]
chunk 2: [48, 55, 63, 71]
chunk 3: [72, 79, 87, 95]
...
```

每组 4 帧覆盖约 24 个控制时刻。action head 会对这些帧做时间重复并增加边界帧，使其匹配 VAE 对一个 block 的时间结构。

客户端参数：

```text
--num-chunks 默认值 = 15
```

这个 15 只是“客户端连续调用多少次模型”，不是 KV cache 大小，也不是 DiT 去噪步数。

## 9. 三条必须分开的时间轴

理解 baseline 最有效的方法，是同时画出三条时间轴：

| 时间轴 | 序列 | 生成/缓存方式 | 当前默认 |
|---|---|---|---|
| 真实机器人时间 | chunk 0、1、2、3... | chunk-level autoregressive | 可连续调用很多次 |
| 单次预测内部 | diffusion step 0...15 | scheduler + DiT flow 复用 | 16 个 scheduler step，8 次 DiT |
| 当前动作 chunk 内部 | action 0...23 | 一次联合去噪 | 不是逐动作 autoregressive |

很多“cache 有几个 chunk”“默认有几步”的混淆，都来自把这三条时间轴当成了同一个序列。

## 10. Causal KV Cache

### 10.1 它缓存什么

Causal KV cache 保存每个 DiT attention layer 对历史 **video token** 计算出的 key/value。

下一次调用时：

- 历史视频 token 不需要重新通过所有 attention layer 计算 K/V；
- 当前 video block 和当前 action/state register 仍然需要完整计算；
- 当前 action/state register 不会作为无限增长的历史 action cache 保存。

因此它更接近：

```text
历史视觉记忆 cache
```

而不是：

```text
保存最近若干条完整动作序列的 LLM KV cache
```

### 10.2 当前 Cache 窗口大小

模型中：

```text
local_attn_size = max_chunk_size × num_frame_per_block + 1
                = 4 × 2 + 1
                = 9 latent frames
```

所以当前 checkpoint 的 causal attention 窗口对应：

```text
初始 frame + 最多 4 个 video/action block
```

这个限制表示模型在一次 causal 上下文中最多使用多少 block，并不表示整个 episode 只能推理 4 次。

### 10.3 Cache 满后会发生什么

当前 action head 在以下情况将 `current_start_frame` 重置为 0：

- 第一次收到语言；
- 语言发生变化；
- 收到单帧 priming 请求；
- `current_start_frame >= local_attn_size`。

因此 demo 可以运行 15 个甚至更多 chunk；达到窗口上限后，模型会使用较新的图像重新建立上下文，而不是结束推理。

当前主要逻辑位于：

```text
groot/vla/model/dreamzero/action_head/wan_flow_matching_action_tf.py
```

## 11. 为什么有人会说默认是 5 个 Chunk

当前仓库中的通用 DROID 数据配置：

```text
groot/vla/configs/data/dreamzero/droid_relative.yaml
```

写的是：

```text
max_chunk_size: 5
```

但是已经训练好的本地 checkpoint：

```text
/home/admin/.cache/DreamZero-DROID/config.json
```

保存的是：

```text
max_chunk_size: 4
```

推理必须以 checkpoint 架构配置为准，所以当前实际值是 4，不是 5。

另外，`socket_test_optimized_AR.py` 虽然声明了 `--max-chunk-size`，但当前 `main()` 没有把该值应用到模型配置。因此截至当前代码，它不能真正覆盖 checkpoint 的 4。

## 12. 单次推理中的 Diffusion

每次预测 action chunk 时，模型创建：

- 未来视频噪声；
- `(24, 32)` 的内部动作噪声。

DROID 对外只使用其中 8 维：

- 7 维 joint position；
- 1 维 gripper position。

其余内部 action register 维度是 padding 或其他 embodiment 的预留空间。

运行时实际使用：

```text
self.num_inference_steps = 16
```

所以一次动作预测包含 16 个 scheduler 积分步骤。checkpoint 中虽然保存了 `num_inference_timesteps=4`，但当前推理函数使用 action head 中设置的 16。

## 13. DiT Flow Cache / DiT Skip Schedule

### 13.1 默认 Static-8

如果没有设置 `NUM_DIT_STEPS`，当前代码默认：

```text
NUM_DIT_STEPS = 8
```

16 个 scheduler step 中，仅以下 step 真正执行 DiT：

```text
[0, 1, 2, 6, 10, 13, 14, 15]
```

其他 step：

- 不执行完整 DiT forward；
- 复用最近一次计算得到的 video flow；
- 复用最近一次计算得到的 action flow；
- video scheduler 和 action scheduler 仍然更新。

所以默认 baseline 实际是：

```text
16 个 scheduler update + 8 个 DiT forward
```

这是一种近似加速，不是数学上完全等价的无损 cache。

### 13.2 支持的固定 Schedule

当前代码显式支持：

| `NUM_DIT_STEPS` | 实际静态 mask |
|---:|---|
| 5 | 5 个指定 step 执行 DiT |
| 6 | 6 个指定 step执行 DiT |
| 7 | 7 个指定 step 执行 DiT |
| 8 | 默认 8 个指定 step 执行 DiT |
| 其他值 | 执行完整 16-call |

因此设置 `NUM_DIT_STEPS=15` 并不会得到 15-call，而会进入 fallback，实际执行 16-call。

5/6/7/8 的 mask 也不嵌套，所以它们不仅调用数不同，调用 DiT 的 timestep 位置也不同。

### 13.3 Dynamic Cache Schedule

设置：

```bash
export DYNAMIC_CACHE_SCHEDULE=true
```

后，`should_run_model()` 会忽略固定 mask，采用 video-flow cosine：

1. 最初两次 DiT 必须执行，用来建立历史；
2. 比较最近两次 video flow 的 cosine similarity；
3. similarity 高于 0.95 时设置更长 skip countdown；
4. 高于 0.93 时设置较短 skip countdown；
5. 被跳过的 step 继续复用最近一次 video/action flow。

当前动态决策只使用 video flow；action flow 虽然已经被保存和复用，但尚未进入相似度 gate。这正是 action-aware 实验希望研究的部分。

### 13.4 DiT Cache 的作用域

DiT flow cache 的 `prev_predictions`：

- 只保留最近两次已计算预测；
- 在每个新的 24-action 推理请求开始时重新创建；
- 不跨 action chunk 保存。

所以它与跨 chunk 的 causal KV cache 完全不同。

## 14. `--enable-dit-cache` 当前是否有效

README 推荐：

```bash
--enable-dit-cache
```

当前服务代码只执行：

```python
os.environ["ENABLE_DIT_CACHE"] = "true" if args.enable_dit_cache else "false"
```

但截至当前工作区：

- 仓库 Python 代码没有读取 `ENABLE_DIT_CACHE`；
- 当前 Conda 环境依赖中也没有找到该环境变量的消费者；
- causal KV cache 本身由 action head 无条件创建和使用；
- DiT 调用次数由 `NUM_DIT_STEPS` 和 `DYNAMIC_CACHE_SCHEDULE` 控制。

因此在当前 PyTorch 推理路径里，`--enable-dit-cache` 实际没有改变行为。README 的该说明与当前实现不同步。

## 15. 两种 Cache 的最终对比

| 项目 | Causal KV cache | DiT flow cache |
|---|---|---|
| 所在时间轴 | 多次机器人策略调用之间 | 一次策略调用的 16 个 diffusion step 内 |
| 缓存内容 | 历史 video attention K/V | 最近一次或两次 video/action flow |
| 是否跨 chunk | 是 | 否 |
| 主要作用 | 不重算历史视频 K/V | 跳过完整 DiT forward |
| 是否近似 | 基本是正常 attention 缓存 | 是，复用旧 flow 会改变去噪轨迹 |
| 当前上限 | checkpoint 对应 4-chunk 窗口 | 最近两次已计算 flow |

可以用下面一句话记忆：

```text
Causal KV cache 减少一次 DiT forward 内的历史计算；
DiT flow cache 减少一整个 DiT forward 被调用的次数。
```

## 16. 两张 GPU 在做什么

当前双卡服务不是 TP、SP，也不是简单把 14B 模型权重各切一半。

代码使用名为 `ip` 的 inference-parallel mesh：

- rank 0 运行正向语言条件；
- rank 1 运行 negative/unconditional 语言条件；
- 两个 rank 交换 video/action 预测；
- video flow 使用 classifier-free guidance 组合条件与无条件结果。

因此每张卡基本都需要容纳一份完整 DiT，这也是本地运行时每卡约占 33.5 GiB 的原因。

Sequence parallelism 若要加入，需要修改 DiT token/attention 的分布式执行方式，不是打开现有双卡开关即可获得。

## 17. 外部动作为什么是 `(24, 8)`

模型内部动作维度为 32，但 DROID action schema 只取：

```text
7 维 joint_position + 1 维 gripper_position
```

DROID checkpoint 对 joint action 使用 relative-action 训练：每个 chunk 的 joint target 相对于该 chunk anchor 的当前关节状态表示。

服务推理后会：

1. 反归一化动作；
2. 将 relative joint action 加回当前 joint state；
3. 返回绝对 joint target；
4. 拼接 gripper，得到 `(24, 8)`。

所以客户端最终拿到的是可由机器人控制接口消费的 24-step action chunk。

## 18. 真实闭环与 Demo 的区别

### 18.1 真实闭环

```text
观察真实机器人
-> 模型预测 24 个动作
-> 执行全部或前 H_exec 个动作
-> 再观察真实机器人
-> 预测下一个 chunk
```

下一次模型调用使用真实的新观测，而不是必须使用模型生成的视频。

### 18.2 `test_client_AR.py`

当前 demo：

- 从 `debug_image/` 读取预录三路视频；
- 按固定 frame schedule 连续发送；
- 记录模型输出但不驱动真实机器人；
- `joint_position`、`cartesian_position` 和 `gripper_position` 填的是零。

因此 demo 的作用是确认：

- checkpoint 能加载；
- WebSocket 和双卡服务能工作；
- 图像 shape、session、KV cache 和输出 shape 正常；
- 显存不会 OOM。

它不能单独证明动作预测质量。做正式 DROID 离线评估时，必须从 Parquet 读取真实 state 和 ground-truth action。

## 19. 当前 Baseline 应如何定义

后续实验应使用以下名称：

### 19.1 官方默认 baseline

```text
causal KV cache
+ static-8 DiT schedule
+ cuDNN SDPA
+ T5 prompt/CPU offload 优化
```

这是没有设置额外 schedule 环境变量时的当前默认路径。

### 19.2 完整计算参考

```text
static-16
```

16 个 scheduler step 全部执行 DiT。但由于固定 mask 不嵌套，static-16 不保证每个样本都比 static-5/8 更准确。

### 19.3 低计算参考

```text
static-5
static-6
```

用于判断减少 DiT 调用会损失多少动作质量。

### 19.4 现有动态 baseline

```text
DYNAMIC_CACHE_SCHEDULE=true
```

只根据 video-flow cosine 动态跳过 DiT。

### 19.5 Action-aware 候选

未来希望比较：

- action-flow convergence；
- provisional action 变化；
- joint 路径、速度、加速度、jerk 和方向变化；
- gripper range、最大单步变化和事件时刻；
- video + action 的组合 gate。

动作特征能够区分抓取阶段，不代表它一定能预测“多算是否有收益”。阶段识别和 compute benefit 必须分别验证。

## 20. 最小心智模型

阅读代码时，可以始终记住下面这张图：

```text
Episode
│
├── Policy call / chunk 0
│     ├── 输入：4 个观测帧 + 当前状态 + prompt
│     ├── 历史：causal video KV cache
│     ├── 内部：16 scheduler step，默认只做 8 次 DiT
│     └── 输出：24×8 action + future video latent
│
├── Policy call / chunk 1
│     ├── 新观测 + 新状态
│     ├── 复用窗口内 video KV
│     └── 再输出 24×8 action
│
├── Policy call / chunk 2
│     └── ...
│
└── causal 窗口达到 checkpoint 的 4-chunk 上限后重新建立上下文
```

一句话总结：

> DreamZero 每次用有限视频历史和当前状态，通过默认 8-call 的 diffusion 共同预测未来世界与 24-step 动作；causal KV cache 管跨 chunk 历史，DiT flow cache 管单次 diffusion 内的跳算，两者不是同一个 cache。

## 21. 关键代码导航

| 内容 | 文件或位置 |
|---|---|
| 服务参数和 WebSocket wrapper | `socket_test_optimized_AR.py` |
| demo 帧序列和默认 15 chunks | `test_client_AR.py` |
| checkpoint 真实结构 | `/home/admin/.cache/DreamZero-DROID/config.json` |
| action head、KV 和 diffusion loop | `groot/vla/model/dreamzero/action_head/wan_flow_matching_action_tf.py` |
| Causal Wan attention 与 KV append | `groot/vla/model/dreamzero/modules/wan_video_dit_action_casual_chunk.py` |
| DROID 三相机拼图 | `groot/vla/model/dreamzero/transform/dreamzero_cotrain.py` |
| DROID 多 chunk 采样 | `groot/vla/data/dataset/lerobot_sharded.py` |
| policy normalization/反归一化 | `groot/vla/model/n1_5/sim_policy.py` |
| 动作难度 pilot | `docs/WAM_ACTION_DIFFICULTY_PILOT.md` |
