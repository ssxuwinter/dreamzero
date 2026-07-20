# WAM 动作难度与动态计算可行性 Pilot 记录

## 1. 文档目的

本文记录 2026-07-19 在本地 DreamZero-DROID 模型和 DROID 数据集上完成的一次小规模探索实验。实验要回答的问题是：

> 能否根据 WAM 输出动作序列的参数大小、变化率、夹爪变化等信号，判断机器人动作处于自由移动还是接触/操作阶段，并进一步决定是否减少或增加 DiT 计算？

这句话实际包含两个必须分开验证的问题：

1. **阶段识别问题**：WAM 输出动作能否区分自由移动、接近、抓取、持物、放置和释放等物理/语义阶段？
2. **计算分配问题**：上述阶段或动作特征能否预测“多执行几次 DiT 会让动作预测更好”？

只有两者同时成立，才可以用动作信号控制动态跳算。能识别抓取阶段，不等于抓取阶段多算一定有收益。

## 2. 结论摘要

本次 pilot 得到四个直接结论：

1. **可以从 WAM 输出中看到阶段信号。** 默认 8-call 输出的夹爪最大单步变化，在本 episode 的 8 个窗口上区分 free/interaction proxy 的样本内 AUC 为 0.9375。
2. **“参数越大、变化越快越难”不成立。** interaction proxy 组的关节最大步长、加速度和 jerk 整体反而更小；机器人在接近接触或精细操作时可能减速并变得更平滑。
3. **尚未证明交互阶段需要更多计算。** 前 8 步 raw joint MAE 上，static-5、static-8、static-16 分别为 0.03087、0.03433、0.03518。16-call 只在 2/8 个窗口小幅优于 5-call，且差异很小。
4. **当前硬件能够运行该实验。** 三个双卡服务可同时运行，每卡约占 33.5 GiB；warmup 后 5/8/16-call 平均约为 2.79/4.00/7.21 秒每 chunk。

因此，本次实验支持“继续研究动作特征”，但不支持现在就实现“接触时多算、自由移动时少算”的在线 gate。

## 3. 要解决的研究问题

### 3.1 现有方法的不足

DreamZero 当前可以使用固定 DiT 调用策略，也可以使用基于 video flow cosine 收敛程度的动态 cache schedule。现有动态策略没有直接使用 WAM action flow 或最终动作几何信息，因此无法回答：

- 视频已经收敛时，动作是否也已经收敛？
- 抓取、持物或释放阶段是否比自由移动阶段更需要计算？
- 是否存在可以安全使用低调用 schedule 的样本子集？
- 动作信号是否比 random gate 和 video-only gate 更有效？

### 3.2 需要避免的循环定义

不能用“动作变化大”直接定义困难，再证明“困难动作变化大”。本项目将三类量分开：

- **阶段代理**：由 ground-truth 夹爪状态构造，只用于解释当前窗口可能处于哪个操作阶段。
- **候选特征**：仅由推理时可获得的 WAM 输出或 action/video flow 构造，用于预测。
- **计算收益**：由同一个样本在不同 schedule 下的动作误差差异得到，是是否值得动态分配计算的目标。

阶段代理不能直接充当 compute-benefit 标签，也不能作为在线 gate 输入。

## 4. 实验环境

### 4.1 代码与软件

| 项目 | 值 |
|---|---|
| 仓库分支 | `main` |
| 基础 commit | `ab790c198fbce33503358efbbd4187ce9a89adf3` |
| 工作区状态 | dirty；本次 pilot 使用了当前工作区中的 T5 CPU offload、cuDNN attention 和服务端改动 |
| Conda 环境 | `/home/admin/miniconda3/envs/dreamzerop` |
| PyTorch | `2.8.0+cu128` |
| CUDA | `12.8` |
| cuDNN | `9.10.2`，PyTorch 版本号 `91002` |
| Attention backend | `cudnn`，通过 PyTorch SDPA 使用 |
| Encoder compile | 关闭，避免 CUDA Graph private-pool 额外显存 |

由于工作区不是 clean commit，本次结果属于探索证据。正式实验必须在 manifest 中保存 commit、dirty diff 摘要和全部推理配置。

### 4.2 硬件

| 项目 | 值 |
|---|---|
| GPU | 8 x NVIDIA RTX PRO 5000 Blackwell |
| 单卡显存 | 48,935 MiB |
| NVIDIA driver | `580.105.08` |
| 单服务拓扑 | 2 张 GPU |
| 三服务并行分配 | 0-1: static-5；2-3: static-8；4-5: static-16 |
| 服务加载后单卡占用 | `nvidia-smi` 约 33.5 GiB |
| 模型内部统计 | 约 32.20 GiB allocated，32.34 GiB reserved |

### 4.3 模型与数据

| 项目 | 路径或规模 |
|---|---|
| DreamZero-DROID checkpoint | `/home/admin/.cache/DreamZero-DROID`，约 43 GiB |
| DreamZero-DROID-Data | `/home/admin/.cache/DreamZero-DROID-Data`，约 131 GiB |
| UMT5 tokenizer | `/home/admin/.cache/umt5-xxl`，约 4.4 MiB |
| T5 CPU offload | 约 10.58 GiB 权重常驻 CPU，执行时按层临时上卡 |
| DiT | 40 层，约 16.48B 参数，双卡推理 |
| 外部动作合同 | `(24, 8)`，7 维绝对关节目标 + 1 维夹爪 |

本次启动时默认 Hugging Face 端点不可达，因此仅从可用镜像下载了 UMT5 tokenizer 的 `config.json`、`tokenizer_config.json`、`spiece.model` 和 `special_tokens_map.json`。模型服务随后以 `HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1` 完全离线启动，没有重新下载模型权重。

## 5. 实验数据与窗口

### 5.1 Episode

本次只使用 DROID episode 0：

- Parquet：`data/chunk-000/episode_000000.parquet`
- 总帧数：237
- 任务文本：`Pick up the blue ring from the table and put it in the wooden tray`
- 成功标记：成功示范
- 相机：两个 exterior camera 和一个 wrist camera
- 状态布局：`observation.state[7:14]` 为 7 维关节位置，`observation.state[6]` 为夹爪位置
- 动作布局：`action[14:21]` 为 7 维绝对关节目标，`action[12]` 为夹爪目标

### 5.2 输入帧协议

每个连续 chunk 使用四帧输入：

```text
[anchor - 23, anchor - 16, anchor - 8, anchor]
```

服务开始时先发送 frame 0 的单帧请求，用来建立该 episode 的 session 和 causal cache。之后按时间顺序发送 8 个窗口，三个 schedule 使用相同输入、状态、任务文本和 session 边界。

| Anchor | 输入帧 | 解释性阶段 | Interaction proxy |
|---:|---|---|---|
| 23 | 0, 7, 15, 23 | 早期自由移动 | 否 |
| 47 | 24, 31, 39, 47 | 自由接近 | 否 |
| 71 | 48, 55, 63, 71 | 后期接近 | 否 |
| 95 | 72, 79, 87, 95 | 夹爪闭合/抓取 | 是 |
| 119 | 96, 103, 111, 119 | 持物移动 | 是 |
| 143 | 120, 127, 135, 143 | 放置前保持 | 是 |
| 167 | 144, 151, 159, 167 | 释放 | 是 |
| 191 | 168, 175, 183, 191 | 释放后移动 | 否 |

### 5.3 Interaction proxy

本次 pilot 使用一个简单且完全机器生成的代理标签：

```text
interaction =
    current_gripper > 0.2
    OR range(future_24_step_gripper) > 0.2
    OR max_abs_step(future_24_step_gripper) > 0.1
```

该规则产生 4 个 free 窗口和 4 个 interaction 窗口。它不是接触力或物体距离真值，只表示夹爪正在切换或处于可能持物的状态。

## 6. 静态 Schedule 配置

模型始终执行 16 个 scheduler 积分步骤，但不同配置只在选定步骤调用 DiT。实际 mask 为：

| 配置 | 调用 DiT 的 step index | 调用数 |
|---|---|---:|
| static-5 | 0, 1, 2, 7, 12 | 5 |
| static-8 | 0, 1, 2, 6, 10, 13, 14, 15 | 8 |
| static-16 | 0-15 全部 | 16 |

这三个 mask 不嵌套。例如 static-5 使用 step 7 和 12，而 static-8 不使用这两个步骤。因此 static-5 与 static-8 的误差差异同时包含“调用数”和“调用位置”两种效应，不能直接解释成纯计算预算效应。

三个服务的关键环境变量和参数保持一致，仅 `NUM_DIT_STEPS` 与 GPU pair 不同：

```bash
CUDA_VISIBLE_DEVICES=0,1 \
NUM_DIT_STEPS=5 \
ATTENTION_BACKEND=cudnn \
COMPILE_ENCODERS=false \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
/home/admin/miniconda3/envs/dreamzerop/bin/torchrun \
  --standalone --nproc_per_node=2 \
  socket_test_optimized_AR.py \
  --port 8105 \
  --model-path /home/admin/.cache/DreamZero-DROID \
  --tokenizer-path /home/admin/.cache/umt5-xxl \
  --attention-backend cudnn \
  --no-compile-encoders
```

static-8 和 static-16 分别使用端口 8108、8116，以及 GPU pair 2-3、4-5。

## 7. 实验过程

1. 确认 8 张 GPU 均为空闲，旧服务进程已经退出。
2. 校验本地 checkpoint、DROID 数据、episode 0 三路视频和 Parquet 动作字段。
3. 补齐并离线加载 UMT5 tokenizer。
4. 并行启动 static-5、static-8、static-16 三个双卡服务。
5. 从服务日志确认：cuDNN attention 生效、T5 CPU offload 生效、输出合同为 `(24, 8)`、各服务实际执行 5/8/16 次 DiT。
6. 每个服务创建独立 session，先发送 frame 0，再按相同顺序发送 8 个四帧窗口。
7. 对每个返回的 24-step 动作计算前 8 步和全 24 步关节/夹爪误差，并从默认 static-8 输出提取动作几何与夹爪特征。
8. 计算 free/interaction 分组均值、样本内 AUC，以及特征与 `E_5-E_16` 的 Spearman 相关。
9. 关闭三个服务，确认 8 张卡均回到 4-6 MiB 空闲占用。

本次 runner 通过终端内联执行，没有生成版本化 `manifest.json` 和逐样本 JSONL。因此本文保存的是聚合和逐窗口结果，不能替代后续正式评估工具产出的机器可读 artifact。

## 8. 指标定义

### 8.1 动作误差

预测与真值按以下时间范围对齐：

```text
prediction[0:24] <-> dataset_action[anchor:anchor+24]
```

本次主观察值使用前 8 步：

```text
joint_mae_h8 = mean(abs(pred_joint[0:8] - gt_joint[0:8]))
gripper_mae_h8 = mean(abs(pred_gripper[0:8] - gt_gripper[0:8]))
```

这里报告的是原始动作单位，没有执行正式 spec 中的不裁剪 q99 缩放。因此数值只用于 schedule 间的同样本探索比较。

### 8.2 动作特征

从当前关节状态、当前夹爪状态和 static-8 的 24-step 预测中计算：

- 关节路径长度与平均/最大单步位移；
- 二阶差分的平均范数，即加速度代理；
- 三阶差分的平均范数，即 jerk 代理；
- 相邻位移方向的变化；
- 预测终点相对当前状态的位移；
- 夹爪 total variation、范围和最大单步变化。

这些特征只使用 WAM 推理输出和当前状态，不使用未来 ground-truth 动作。

### 8.3 阶段区分指标

AUC 由 4 个 interaction 与 4 个 free 样本做成对比较得到。由于只有一个 episode，所有 AUC 都是样本内诊断值，不是泛化性能。

## 9. 实验结果

### 9.1 可运行性和显存

- 三个双卡服务可以同时加载并提供推理，没有发生 OOM。
- 每张被使用的 GPU 在服务空闲时约占 33.5 GiB，距离 48.9 GiB 上限仍有约 15 GiB。
- T5 约 10.58 GiB 常驻 CPU，按层临时上卡；DiT、CLIP 和 VAE 常驻 GPU。
- 本次没有使用 sequence parallelism。对于离线样本扫描，独立双卡服务可以直接利用 8 卡并行多个 schedule。

### 9.2 延迟

| Schedule | 初始单帧请求 | 首个四帧请求 | 后续 7 个窗口平均 | 后续窗口中位数 |
|---|---:|---:|---:|---:|
| static-5 | 9.51 s | 26.86 s | 2.79 s | 2.82 s |
| static-8 | 10.43 s | 26.75 s | 4.00 s | 3.91 s |
| static-16 | 12.83 s | 29.30 s | 7.21 s | 7.17 s |

首个四帧请求包含约 21-22 秒的一次性 scheduler 或 kernel 初始化开销，不能混入 steady-state latency。进入稳态后，static-16 约为 static-5 的 2.59 倍，static-8 约为 static-5 的 1.44 倍。

### 9.3 聚合动作误差

| Schedule | Joint MAE H8 全部 | Free | Interaction | Gripper MAE H8 全部 |
|---|---:|---:|---:|---:|
| static-5 | **0.03087** | **0.03955** | **0.02219** | **0.05076** |
| static-8 | 0.03433 | 0.04451 | 0.02415 | 0.05136 |
| static-16 | 0.03518 | 0.04685 | 0.02351 | 0.05273 |

在这个 episode 上，交互组 joint MAE 更低，而不是更高。这主要说明交互阶段动作较慢、短期轨迹更容易贴近单条示范，不代表物理任务更简单。

### 9.4 逐窗口 Joint MAE H8

| Anchor | 阶段 | static-5 | static-8 | static-16 | 当前最小值 |
|---:|---|---:|---:|---:|---|
| 23 | 早期自由移动 | **0.04717** | 0.04982 | 0.05406 | static-5 |
| 47 | 自由接近 | **0.02906** | 0.04019 | 0.04832 | static-5 |
| 71 | 后期接近 | 0.04238 | 0.04557 | **0.04189** | static-16 |
| 95 | 夹爪闭合 | **0.01558** | 0.01825 | 0.01630 | static-5 |
| 119 | 持物移动 | **0.01046** | 0.01220 | 0.01350 | static-5 |
| 143 | 放置前保持 | 0.02357 | 0.02631 | **0.02312** | static-16 |
| 167 | 释放 | **0.03915** | 0.03983 | 0.04111 | static-5 |
| 191 | 释放后移动 | **0.03958** | 0.04247 | 0.04313 | static-5 |

static-5 在 6/8 个窗口上误差最小。static-16 在另外两个窗口上的优势分别只有约 0.00050 和 0.00044，当前样本量下不能视为稳定的高计算收益。

### 9.5 Static-8 输出特征与阶段代理

| 特征 | Free 均值 | Interaction 均值 | 最佳方向样本内 AUC | 方向 |
|---|---:|---:|---:|---|
| Joint 最大单步位移 | 0.05418 | 0.03902 | 0.7500 | 较小更像 interaction |
| Joint 加速度代理 | 0.01012 | 0.00906 | 0.8125 | 较小更像 interaction |
| Joint jerk 代理 | 0.01620 | 0.01449 | 1.0000 | 较小更像 interaction |
| Gripper total variation | 0.23438 | 0.56249 | 0.8125 | 较大更像 interaction |
| Gripper 范围 | 0.21094 | 0.48722 | 0.8125 | 较大更像 interaction |
| Gripper 最大单步变化 | 0.02661 | 0.14901 | **0.9375** | 较大更像 interaction |

该表说明：

- 夹爪输出是很自然的抓取/释放阶段信号。
- 关节特征也有阶段信息，但方向与“越大越难”的直觉相反。
- 单 episode 中 jerk 的 AUC=1.0 很可能包含明显的轨迹阶段和样本选择效应，不能作为最终模型指标。
- release 后窗口仍可能预测较大的夹爪变化，因此简单阈值会产生阶段滞后和假阳性。

### 9.6 特征与额外计算收益

以 `E_5-E_16` 表示 16-call 相对 5-call 的 joint MAE 改善时：

- gripper 最大单步变化与该收益的 Spearman 相关约为 0.30；
- gripper range 约为 0.17；
- joint jerk 约为 -0.55；
- 样本数只有 8，且 static-5/static-16 mask 不嵌套，不能对这些相关值做统计结论。

最重要的观测是：能很好区分 interaction 的夹爪特征，并没有同时表现出足够强的计算收益预测力。这正是“阶段识别”和“计算分配”必须分开的原因。

## 10. 对原始想法的判定

### 10.1 能否根据动作参数判断动作阶段？

**初步可以。** 夹爪最大变化、范围和 total variation 能识别闭合、持物和释放窗口；关节减速、较小 jerk 等信号也能辅助描述精细操作阶段。

### 10.2 能否使用“动作越大、变化越快”判断难度？

**不可以使用单调规则。** 抓取和接触附近经常需要减速，动作可能更小、更平滑。至少需要联合使用：

- 夹爪事件；
- 速度与减速模式；
- 方向变化和尺度无关 roughness；
- action flow/provisional action 的收敛信号；
- 可选的视觉距离或接触估计，但这超出 action-only v1。

### 10.3 能否据此决定跳算？

**目前不能。** 本次没有观察到 interaction 窗口从 static-16 获得稳定收益，static-5 反而在多数窗口上更准。必须先在更多 episode 上证明 compute benefit 存在稳定异质性，再训练或设定 gate。

### 10.4 是否需要人工标注？

第一阶段不需要大规模人工难度标注。可以使用 ground-truth 夹爪轨迹生成 free、pre-close、closing/contact-proxy、hold、release 代理，用于分层分析。

如果最终研究目标要求证明“机械爪距离物体多远”或“是否真实接触”，仅靠动作序列不够，需要物体位姿、接触力、视觉距离估计或少量人工审查。人工标签应只验证阶段代理，不应定义 compute benefit。

## 11. 本次 Pilot 的限制

1. 只有一个 episode 和 8 个窗口，AUC 与相关性没有独立 test 泛化意义。
2. interaction proxy 来自夹爪状态，不是真实物体距离或接触传感器标签。
3. 报告的是 raw action MAE，没有使用正式的逐关节 q99 缩放。
4. 5/8/16-call step mask 不嵌套，schedule 位置与调用数效应混杂。
5. DROID 是成功示范数据，离线 GT action error 不等同于闭环任务成功率。
6. 单条示范存在多模态问题：不同但同样成功的动作可能被 raw MAE 惩罚。
7. 本次未运行 static-6 和 video-only 配对组。
8. 服务使用模型内部默认噪声生成路径，但本次没有做重复运行来验证 bitwise reproducibility。
9. 首个四帧窗口有明显一次性开销；不分 cold/warm 会严重扭曲延迟结论。
10. 工作区为 dirty 状态，内联 runner 没有保存完整 manifest 和逐样本 JSONL。

## 12. 下一步正式验证方案

### 12.1 Phase 0：先建立可信评估器

- 固定 episode-level train/validation/test split。
- 保存 sample manifest、完整 step mask、seed、checkpoint、代码版本和 dirty diff。
- 明确 `H_exec`；当前建议先用 8，并同时报告 H=1/6/8/24 敏感性。
- 使用不裁剪的 checkpoint q99 缩放，同时保留 raw-radian 指标。
- 验证连续 episode 的 priming、session、reset 和 causal/KV-cache 完全配对。
- 重复固定样本，估计随机噪声并冻结三档 tolerance。

### 12.2 Phase 1：验证计算收益是否真的存在

- 分层诊断集：50 个 episode，每个最多 10 个非重叠 anchor，约 500 个样本每配置。
- 自然分布确认集：从未参与定参的 episode 按自然阶段占比抽样。
- 并行运行 static-5/6/8/16，随后追加 video-only。
- 计算可接受静态 schedule、成对 compute benefit、P90/P95 和 episode bootstrap CI。
- 如果高调用 schedule 没有稳定改善任何具有足够样本量的子集，则 H1 no-go，停止 action-aware gate 实现。

### 12.3 Phase 2：验证动作信号是否能预测计算收益

特征分为：

- `V`：video flow 收敛；
- `A`：action flow cosine、相对 L2、provisional action 变化；
- `M`：动作几何、减速、roughness、夹爪事件；
- `A+M`：上述可解释组合。

阈值、逻辑回归或浅层树只能在 train/validation 上拟合。只会区分 phase、不能预测 compute benefit 的特征不得进入 gate。

### 12.4 Phase 3：真实在线回放

静态 schedule 结果不能证明真实跳算有效。最多锁定两个候选 gate，在同一个 16-step 去噪轨迹中实际执行 run/skip，再与 always-8、matched-compute random 和 video-only 比较。

建议保持现有 go 条件：

- 相对 always-8，平均 DiT 调用减少至少 20%；
- 主指标均值退化不超过 2%；
- P95 与 contact-proxy 子集退化不超过 5%；
- 在匹配计算量下优于 random 和 video-only；
- action-aware 模式默认关闭，异常时 fail-compute。

## 13. 最终回答

对于“能否根据 WAM 输出动作序列参数判断动作难易，并决定是否跳过计算”这一要求，当前结论是：

- **阶段判断：有初步可行性。** 尤其是夹爪变化和关节减速/平滑度。
- **简单大小阈值：不可行。** 接触附近可能更慢、更平滑。
- **计算跳过：尚未验证。** 本次甚至观察到 static-5 在多数窗口上优于 static-16。
- **实验平台：可行。** 8 x RTX PRO 5000 足以并行开展正式配对实验。

因此正确的研究路径不是立刻写 gate，而是先用更多 episode 验证“额外计算收益是否存在异质性”，再判断 WAM action-flow 或动作参数能否预测这种收益。

## 14. 对应 OpenSpec

本实验的后续规范与任务记录位于：

- `openspec/changes/action-aware-dynamic-dit-scheduling/proposal.md`
- `openspec/changes/action-aware-dynamic-dit-scheduling/design.md`
- `openspec/changes/action-aware-dynamic-dit-scheduling/specs/adaptive-dit-evaluation/spec.md`
- `openspec/changes/action-aware-dynamic-dit-scheduling/specs/action-aware-dit-scheduling/spec.md`
- `openspec/changes/action-aware-dynamic-dit-scheduling/tasks.md`

OpenSpec 将“物理阶段代理”“静态 schedule compute benefit”和“真实在线 skip 收益”明确分开，避免把本次单 episode 的阶段信号误写成已经证明动态计算有效。
