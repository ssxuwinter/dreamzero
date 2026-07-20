# WAM 动作阶段识别与动态计算实验报告

## 1. 一句话结论

**动作输出可以识别机器人何时从自由移动进入精细操作，但当前证据不支持用这个阶段信号决定“精细时多算、自由时少算”。**

更具体地说：

- 师兄关于“长动作序列存在简单前段与精细后段”的直觉成立。
- “动作幅度更大、加速度/jerk 更大就更难”不成立；精细阶段往往更慢、更平滑。
- 阶段识别和模型计算难度是两个不同问题。前者可识别，后者在当前 DreamZero-DROID 离线实验中不可预测。
- 不建议现在实现 phase-aware DiT gate。若目标是加速，下一步应先闭环验证固定 2/4/8-call，而不是假定精细阶段需要 16-call。

## 2. 师兄的想法到底是什么

假设一条完整机器人动作有 100 步：

```text
0 ------------------------------------------------------ 100
|       自由空间移动 / 巡航       | 接近 | 接触/抓取/调整 |
|          相对简单               | 过渡 |     精细       |
```

这个想法包含两个独立命题：

1. **阶段识别命题**：能否从动作序列的速度、加速度、jerk、方向、夹爪变化和相邻 chunk 变化看出机器人已经进入精细阶段？
2. **计算分配命题**：进入精细阶段时，WAM 继续调用更多 DiT 是否真的能改善动作预测？

只有两个命题都成立，才能推出：

```text
自由移动 -> 少算
精细操作 -> 多算
```

本项目的核心发现是：**第一个成立，第二个不成立。**

## 3. DreamZero 中的动作是什么

本地 DreamZero-DROID 每次返回一个 `(24, 8)` 动作 chunk：

- 24：未来 24 个控制步；
- 前 7 维：绝对关节目标；
- 第 8 维：夹爪目标。

模型内部 action register 有 32 维，但 DROID 只使用 8 个有效维，其余 padding 维不能进入动作难度或 flow 指标。

长轨迹不是一次输出 100 步，而是连续输出多个 24-step chunk。我们同时分析：

- chunk 内：路径、速度、加速度、jerk、方向变化、roughness、夹爪事件；
- chunk 间：边界跳变、速度/加速度统计量变化、运动方向变化、夹爪状态变化；
- 去噪过程中：action flow cosine、相对 L2、prefix action 收敛。

## 4. 时间因果约束

必须区分两种 gate：

```text
前一个 chunk 的最终输出  -> 只能决定下一个 chunk 怎么算
当前 chunk 的中间输出    -> 才能决定当前请求是否继续算
```

因此：

- 最终动作参数适合做 `lagged next-chunk scheduling`；
- 当前请求早停必须看 provisional action 或 action flow；
- 用当前 chunk 完成后的最终动作回头解释“本 chunk 可以少算”是不成立的。

## 5. 实验问题与预注册假设

| 假设 | 内容 | 最终状态 |
|---|---|---|
| H1 | predicted gripper 和 chunk transition 可识别精细阶段 | 支持 |
| H2 | 动作幅度/速度/加速度/jerk 不与精细难度正单调 | 支持 |
| H3 | 语义精细阶段不足以预测额外 DiT 收益 | 支持 |
| H4 | provisional/action-flow 收敛可预测继续计算收益 | 当前指标下否定 |

所有协议在读取确认结果前保存于 `research/action_chunk_compute/experiments/`，并使用 episode 作为交叉验证和 bootstrap 单位，避免同一轨迹的相邻窗口泄漏。

## 6. 实验环境

| 项目 | 配置 |
|---|---|
| GPU | 8 x NVIDIA RTX PRO 5000 Blackwell，单卡约 48.9 GiB |
| 模型 | `/home/admin/.cache/DreamZero-DROID` |
| 数据 | `/home/admin/.cache/DreamZero-DROID-Data`，57,774 episode |
| 文本编码器 | UMT5-XXL 常驻 CPU、逐层临时上 GPU |
| Attention | PyTorch SDPA 的 cuDNN backend |
| 单服务拓扑 | 2 GPU |
| 动作 horizon | 24 |
| 主评价 horizon | H8 |
| 关节指标 | q99 range 逐关节缩放、不裁剪 MAE |

三组静态服务可同时运行。稳态延迟约为：

| DiT calls | 延迟/chunk |
|---:|---:|
| 5 | 2.70 s |
| 8 | 3.92 s |
| 16 | 7.20 s |

## 7. 实验 A：GT 动作是否包含阶段信号

### 7.1 数据与标签

- 随机种子：`20260719`；
- 1,200 个成功 episode；
- 11,030 个连续非重叠 24-step chunk；
- 10,674 个清晰阶段 chunk，356 个 ambiguous；
- 阶段：`free_open`、`pre_close`、`closing`、`hold`、`release`。

标签来自 GT 夹爪迟滞状态，只用于离线分层，不是真实接触力或物体距离，也不作为在线 gate 输入。

### 7.2 阶段识别结果

| 任务 | 特征 | AUROC | AUPRC |
|---|---|---:|---:|
| fine vs free | joint-only | 0.662 | 0.858 |
| fine vs free | gripper-only | 0.942 | 0.983 |
| fine vs free | joint+gripper+transition | 0.955 | 0.987 |
| pre-close vs free | joint-only | 0.687 | 0.546 |
| pre-close vs free | gripper-only | 0.699 | 0.665 |
| pre-close vs free | joint+gripper | 0.787 | 0.738 |
| pre-close vs free | joint+gripper+transition | **0.800** | **0.747** |

对 `pre-close vs free`，transition 相对 joint+gripper 的独立 AUROC 增量为 0.0132，episode bootstrap 95% CI `[0.0038, 0.0225]`。这说明相邻 chunk 确实包含小但稳定的阶段边界信息。

### 7.3 “越大越难”为什么不成立

代表性中位数：

| 特征 | free | pre-close | closing |
|---|---:|---:|---:|
| joint mean step | 0.01571 | 0.01183 | 0.00887 |
| joint max step | 0.05757 | 0.05336 | 0.03783 |
| joint acceleration | 0.00740 | 0.00574 | 0.00465 |
| joint jerk | 0.01029 | 0.00791 | 0.00628 |

机器人接近物体和闭合夹爪时通常减速，动作变小且更平滑。因此正确描述不是“变化越大越困难”，而是联合观察：

- 速度是否下降；
- 夹爪是否即将切换；
- 方向和 endpoint 是否改变；
- 相邻 chunk 的统计量是否发生阶段性变化。

## 8. 实验 B：WAM 输出能否识别阶段，并预测静态计算收益

### 8.1 设计

- 12 个未使用 episode；
- 每个 episode 7 个连续 chunk，共 84 个；
- 每条轨迹覆盖 free、pre-close、closing、hold、release；
- 三个独立双卡服务对完全相同输入运行 static-5/8/16；
- 固定 seed 重复子集逐元素差为 0。

### 8.2 质量和延迟

| Calls | H8 joint MAE | H24 joint MAE | H8 gripper MAE | 延迟 |
|---:|---:|---:|---:|---:|
| 5 | **0.02108** | 0.03437 | 0.10422 | 2.70 s |
| 8 | 0.02121 | **0.03389** | **0.09177** | 3.92 s |
| 16 | 0.02174 | 0.03506 | 0.09226 | 7.20 s |

`E5-E16` 的均值为 -0.00066，95% CI `[-0.00167, 0.00049]`。负值表示 16-call 在主指标上没有改善。

### 8.3 阶段识别与计算收益预测分离

WAM predicted-gripper 对 fine/free 的 episode-held-out AUROC 为 0.911，说明模型输出包含明显阶段信号。

但是预测 `benefit>0.002` 时：

| 特征 | AUROC | R2 |
|---|---:|---:|
| joint | 0.452 | -0.157 |
| gripper | 0.429 | -0.043 |
| joint+gripper+transition | 0.461 | -0.176 |
| lagged previous chunk | 0.388 | -0.306 |
| GT phase oracle | 0.327 | -0.065 |

这就是本项目最重要的反例：**同一个特征可以很好地识别“正在抓取”，但完全不能判断“多算会不会更准”。**

## 9. 实验 C：同一去噪轨迹的因果 prefix-stop

### 9.1 为什么还需要这个实验

static-5/8/16 的 step mask 不嵌套，差异同时包含调用次数和调用位置。为排除这个混杂，我们运行完整 16-call，并保存每一步 8 个有效动作维上的 action flow。

对每个 `k=1..16`：

1. 使用与 full-16 完全相同的前 k 个 flow；
2. k 之后不再调用 DiT；
3. 复用第 k 个 flow 完成剩余 scheduler 积分；
4. 得到 prefix-k 动作。

prefix-16 与正常 final 的最大逐元素差为 0，也与独立 static-16 输出完全一致。

### 9.2 Quality-compute 曲线

| Calls | H8 joint MAE | 95% CI | 预测运动量 H8 |
|---:|---:|---:|---:|
| 1 | 0.01818 | [0.01552, 0.02158] | 0.01518 |
| 2 | **0.01817** | [0.01554, 0.02146] | 0.01221 |
| 4 | 0.01946 | [0.01652, 0.02323] | 0.01230 |
| 8 | 0.02119 | [0.01821, 0.02489] | 0.01362 |
| 12 | 0.02169 | [0.01865, 0.02586] | 0.01347 |
| 16 | 0.02174 | [0.01860, 0.02603] | 0.01345 |

Persistence baseline 为 0.02475，GT motion H8 为 0.01595。prefix-1 并非完全不动，但离线单示范 MAE 仍可能偏好更保守的预测。

### 9.3 精细阶段是否更需要多算

这里 `benefit = E_k-E16`，正数才表示继续到 16-call 有帮助。

| 阶段 | 2→16 | 4→16 | 8→16 |
|---|---:|---:|---:|
| free | -0.00185 | -0.00111 | -0.00103 |
| pre-close | -0.00286 | -0.00162 | **0.00003** |
| closing | -0.00340 | -0.00292 | -0.00101 |
| hold | -0.00420 | -0.00265 | -0.00058 |
| release | -0.00573 | -0.00402 | -0.00042 |

没有一个精细阶段表现出稳定的额外计算收益。pre-close 的 8→16 收益约 0.00003，可视为零。

即使只看 GT 运动量最高的三分之一样本，2→16 mean benefit 仍为 -0.00083。

### 9.4 flow 或 provisional convergence 能否预测收益

episode-held-out 结果：

- k=2 最佳 AUROC 0.597，最佳 R2 0.042；
- k=4 最佳 AUROC 0.609，R2 仍不超过约 0；
- k=8 最佳 AUROC 0.557，R2 为负；
- GT phase oracle 的 AUROC 为 0.421/0.486/0.480。

这些数值不足以构建 gate，也没有优于可靠基线的证据。

## 10. 最终回答

### 10.1 能否识别“前 80 步简单、后 20 步精细”？

**可以。** 但不是靠“越大越难”，而是通过夹爪事件、关节减速、动作几何和 chunk transition 联合识别。

### 10.2 能否直接用加速度或变化率设阈值？

**不可以。** 精细阶段常常加速度和 jerk 更小。单调阈值会把快速自由运动误判为难，把慢速接触误判为简单。

### 10.3 能否据此让精细阶段多算？

**当前不能。** 三组实验均未显示精细阶段从更多 DiT 获得稳定收益，held-out predictor 也没有预测力。

### 10.4 这个 idea 是完全没用吗？

不是。阶段识别本身可用于：

- 轨迹自动分段和数据分析；
- 区分 approach/grasp/hold/release；
- 调整机器人执行多少个动作后重新观测，即 adaptive execution chunk；
- 为少量视觉/力觉接触标签挑选候选片段。

这些用途不等于动态 DiT 计算。

## 11. 为什么不能宣称 1-2 call 已经最好

DROID 只有成功示范，每个观测只对应一条记录动作。机器人任务通常多模态：另一条与 GT 不同的动作也可能成功。离线 MAE 可能：

- 惩罚不同但合理的动作；
- 偏好小幅、保守或接近当前姿态的动作；
- 无法衡量接触稳定性、碰撞、安全性和最终任务成功。

所以本报告可以否定“阶段感知多算已有证据”，但不能用 MAE 单独证明 1-call 的机器人成功率优于 8/16-call。

## 12. 建议的下一步

1. 不实现 phase-aware more-compute gate，保持默认路径不变。
2. 在仿真或真机闭环上直接比较固定 2/4/8-call：成功率、安全性、动作平滑度、失败恢复和延迟。
3. 若固定低预算闭环不退化，优先采用简单固定预算；它比学习 gate 更可靠。
4. 若确实要识别真实接触，增加视觉物体距离、末端位姿或力/触觉信号，并用少量人工审核验证代理标签。
5. 只有闭环结果证明某些状态确实从额外计算受益，才重新训练 compute-benefit gate。

## 13. 可复现实验产物

| 产物 | 路径 |
|---|---|
| 研究状态 | `research/action_chunk_compute/research-state.yaml` |
| 文献定位 | `research/action_chunk_compute/literature/survey.md` |
| H1 协议/结果 | `research/action_chunk_compute/experiments/h1_gt_chunk_phase/` |
| H2 协议/结果 | `research/action_chunk_compute/experiments/h2_wam_static_compute/` |
| H3 协议/结果 | `research/action_chunk_compute/experiments/h3_nested_provisional_compute/` |
| GT 分析器 | `research/action_chunk_compute/src/analyze_gt_chunk_phase.py` |
| 静态配对 runner/分析器 | `research/action_chunk_compute/src/run_wam_static_pairing.py`、`analyze_wam_static_pairing.py` |
| prefix trace runner/分析器 | `research/action_chunk_compute/src/run_wam_prefix_trace.py`、`analyze_wam_prefix_trace.py` |

## 14. 相关一手工作

- [DreamZero: World Action Models are Zero-shot Policies](https://arxiv.org/abs/2602.15922)：WAM 联合预测未来世界状态和动作。
- [Diffusion Policy](https://arxiv.org/abs/2303.04137)：动作扩散和 receding-horizon control。
- [ACT](https://arxiv.org/abs/2304.13705)：精细操作依赖精度、接触协调和闭环反馈。
- [Adaptive Action Chunking at Inference-time](https://arxiv.org/abs/2604.04161)：动作不确定性可调节执行 chunk 长度，但不证明应调节 DiT 调用数。
- [A Simple Early Exiting Framework for Accelerated Sampling in Diffusion Models](https://arxiv.org/abs/2408.05927)：扩散计算可早退，但依据扩散时间/网络计算，不等于机器人语义阶段门控。

