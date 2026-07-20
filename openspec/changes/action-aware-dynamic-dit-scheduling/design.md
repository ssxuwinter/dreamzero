## Context

### 研究问题

师兄提出的直觉包含两个不同命题：

1. **物理/语义难度**：机械臂在自由空间移动时较简单，接近物体、闭合夹爪、接触或保持物体时更困难。
2. **计算难度**：某些 WAM 样本需要更多 DiT 调用才能得到足够好的动作，而另一些样本可以提前停止或复用 flow。

只有同时验证“动作信号能描述关键阶段”和“这些阶段或信号能预测额外计算收益”，才能据此动态分配计算。动作幅度、速度或变化率可以作为特征，但不能直接充当难度真值；特别是接触前动作往往会减速，因此“小且平滑”也可能对应关键阶段。

### 当前模型与调度机制

- DROID 服务对外返回 `(24, 8)` 动作序列：前 7 维为绝对关节目标，第 8 维为夹爪；训练时关节动作使用相对当前位置的表示，推理后处理再加回当前状态。
- 模型内部 action register 为 32 维，DROID 未使用的 padding 维度不能进入 action-aware 指标。
- 一次推理始终有 16 个 scheduler 积分步骤；当前默认 `NUM_DIT_STEPS` 对应 8 次实际 DiT 调用。5、6、7、8-call 使用各自的静态 step mask，其他值执行完整 16-call。
- 这些静态 mask 并不嵌套，所以“先完成 5-call，再无损继续到 16-call”并不成立。真正的在线自适应应在单次 16-step 轨迹中逐步决定 run/skip。
- 现有 `DYNAMIC_CACHE_SCHEDULE=true` 始终执行最初两次 DiT，然后仅比较相邻 video flow 的 cosine similarity；超过 0.95 时设置 4-step countdown，超过 0.93 时设置 2-step countdown。
- 每次实际 DiT 调用已经同时得到 `flow_pred_video` 和 `flow_pred_cond_action`，并保存在 `prev_predictions` 中。因此 action-flow 信号无需增加一次模型前向即可获得。
- 跳过 DiT 时，现有实现会复用最近一次 video/action flow，但 video 与 action scheduler update 仍执行。

### 本地数据、模型与硬件约束

- checkpoint：`/home/admin/.cache/DreamZero-DROID`。
- 数据集：`/home/admin/.cache/DreamZero-DROID-Data`；本地清点为 57,774 个 episode、14,748,517 帧，15 FPS，平均约 255.28 帧/episode。
- 转换后的 episode 全部标记为成功，但没有 frame-level 接触、物体距离或动作难度标签。因此 v1 不依赖人工难度标注，ground-truth 夹爪阶段只能作为接触代理。
- 每个 chunk 覆盖 24 个控制步。当前 demo 以 24 帧为 anchor 间隔，并使用 `[-23, -16, -8, 0]` 四个输入帧偏移；实验 manifest 必须保存实际对齐规则。
- 机器有 8 张 RTX PRO 5000，每张约 48 GiB。当前可运行配置每个服务占 2 张 GPU，T5 常驻 CPU、按层临时上卡，默认使用 `ATTENTION_BACKEND=cudnn` 的 PyTorch SDPA cuDNN backend。

### 已有探索证据

以下结果只用于确定实验是否值得做，不能替代正式统计结论：

| 探索项 | 观测 |
|---|---|
| 1,200 个 episode 的数据统计 | stable-open/free 的关节速度中位数约 0.03631，pre-close 约 0.02524；对应 jerk 中位数约 0.02288 与 0.01533 |
| 含义 | 接近夹取时反而更慢、更平滑，“幅度或变化率越大越难”这一单调规则不成立 |
| episode 0，static-5 | 稳态约 2.75 s/chunk；原始 joint MAE 约 0.06290，gripper MAE 约 0.10347 |
| episode 0，默认 static-8 | 稳态约 3.7-4.4 s/chunk；原始 joint MAE 约 0.06408，gripper MAE 约 0.10752 |
| episode 0，static-16 | 稳态约 7.14 s/chunk；原始 joint MAE 约 0.06789，gripper MAE 约 0.09981 |
| episode 0，现有 video-only 动态调度 | 10 个 chunk 全部使用 4 次 DiT，稳态约 2.1-2.6 s/chunk；原始 joint MAE 约 0.05923，gripper MAE 约 0.10135 |
| episode 0，8 个连续窗口的 5/8/16-call 配对 pilot | 前 8 步 raw joint MAE 分别为 0.03087、0.03433、0.03518；16-call 仅在 2/8 个窗口小幅优于 5-call，尚无“交互阶段需要更多计算”的证据 |
| 同一 pilot 的默认 8-call 动作特征 | predicted-gripper 最大单步变化区分 free/interaction proxy 的样本内 AUC 为 0.9375；interaction 组关节 jerk 均值反而更低（0.01449 对 0.01620） |
| 同一 pilot 的资源与延迟 | 每卡约占 33.5 GiB；去掉首个四帧 warmup 窗口后，5/8/16-call 平均约 2.79/4.00/7.21 s/chunk |

单 episode 上，动作参数已经表现出阶段信号，但更多计算没有单调改善关节 MAE。该结果只支持“可以继续验证动作特征”，不支持“交互阶段应该多算”或“可以立即接入在线 gate”。现有 video-only gate 虽然快，但在 free、grasp、release chunk 上都给出相同的 4-call，尚未显示语义区分能力。正式实验必须使用归一化相对动作指标、更多 episode、自然分布确认集和 episode 配对置信区间。

### 最终阶段富集诊断证据（2026-07-19）

| 实验 | 样本 | 关键结果 | 判定 |
|---|---:|---|---|
| GT chunk phase | 1,200 episode，10,674 个清晰 chunk | pre-close vs free AUROC：joint 0.687、gripper 0.699、joint+gripper+transition 0.800 | 阶段信号成立 |
| WAM static pairing | 12 episode x 7 连续 chunk | 5/8/16-call H8 joint MAE：0.02108/0.02121/0.02174 | 更多静态调用无总体收益 |
| WAM phase classifier | 同上 | predicted-gripper fine/free AUROC 0.911 | WAM 输出含阶段信号 |
| 静态 compute predictor | 同上 | benefit>0.002 AUROC 0.388-0.461；GT phase oracle 0.327 | 阶段信号不能预测收益 |
| 同轨迹 prefix-stop | 同上，k=1..16 | 1/2/4/8/16-call H8 joint MAE：0.01818/0.01817/0.01946/0.02119/0.02174 | 因果前缀不支持多算 |
| prefix compute predictor | 同上 | 最佳 AUROC 约 0.609，R2 基本为 0 或负；GT phase oracle 无效 | action flow/几何均 no-go |

closing 的关节平均步长、加速度和 jerk 均低于 free，明确反驳“大/快/变化剧烈等于难”。所有精细阶段的 `E2-E16` 均为负；pre-close 的 `E8-E16` 约 0.00003，可视为零。

这批样本是阶段富集诊断集，不替代自然分布或闭环成功率确认。但它已经满足停止条件：连 GT phase oracle 都不能预测额外计算收益，因此不再拟合或接入在线 action-aware gate，也不追加 video-only matched-compute 比较。完整结果见 `docs/2_action_phase_compute_research.md`。

### 可证伪假设

- **H1：计算收益异质性。** 固定 5/6/8/16-call 的配对结果中，存在稳定的一组样本可低预算完成，也存在一组样本从额外 DiT 调用中显著受益。
- **H2：动作信号具有增量预测力。** 决策时刻可获得的 action-flow 收敛或 provisional action 特征，比同计算量 random gate 和现有 video-only 相似度更能预测静态 schedule compute benefit，并能在真实在线回放中转化为更好的质量-计算量折中。
- **H3：在线收益可兑现。** 在真实 run/skip 去噪轨迹中，action-aware gate 相对默认 8-call 至少减少 20% 平均 DiT 调用，同时主指标均值退化不超过 2%，P95 与 contact-proxy 子集退化不超过 5%。

最终状态：语义阶段识别假设得到支持；H1 的“高预算稳定收益”和 H2 的“动作信号预测 compute benefit”均未得到支持；H3 因前置门失败而不进入在线 adaptive-gate 实现。

H1 失败时停止 gate 研究；H1 成立但 H2 失败时保留预算扫描工具、停止 action-aware 实现；只有 action-flow 成立而物理动作特征失败时，可以研究“收敛调度”，但不能宣称验证了“接触动作更难”。

## Goals / Non-Goals

**Goals:**

- 建立从固定 schedule 配对推理、可接受 schedule/compute-benefit 构造、特征分析到在线 gate 回放的完整实验链。
- 将语义阶段代理与计算收益标签分离，避免循环定义。
- 在关节、夹爪、平均误差、尾部误差和关键阶段上公平比较各调度策略。
- 使用已有 action flow，以极小的 gate 开销实现单次在线动态跳算。
- 所有新增行为显式 opt-in，现有客户端、默认 8-call 和 video-only 路径不受影响。

**Non-Goals:**

- v1 不训练新的 WAM、DiT 或动作预测网络。
- v1 不把 ground-truth 夹爪阶段作为在线输入，也不使用未来帧或真值动作作 gate 特征。
- v1 不声称仅靠关节动作就能测得机械爪到物体的真实距离或真实接触。
- v1 不以生成视频的视觉质量作为主目标，也不默认解码和保存所有预测视频。
- v1 不引入 sequence parallelism；四个独立双卡服务更适合本次离线样本并行。
- v1 不用单一 raw MAE 或单一 episode 得出最终结论。

## Decisions

### 1. 用静态 schedule 充分性和 compute benefit 定义目标，而不是人工 easy/hard 标签

对每个样本、每个固定 schedule `s in {5, 6, 8, 16}` 计算主误差 `E_s`：

```text
E_best = min(E_5, E_6, E_8, E_16)
acceptable_static_schedules = { s | E_s <= E_best + tolerance }
lowest_call_acceptable_schedule = argmin_calls(acceptable_static_schedules)
compute_benefit(low, high) = E_low - E_high
```

这样允许 schedule-质量关系非单调，也不假设 16-call 一定最好。tolerance 至少取严格、中等、宽松三档做敏感性分析；具体默认值由 pilot 中主指标的尺度和重复推理噪声确定，并写入 manifest。

5、6、8、16-call 使用不同且不嵌套的 step mask，因此 `lowest_call_acceptable_schedule` 只表示候选静态策略中的最低调用数可接受项，不能解释为“同一去噪轨迹至少需要多少次计算”。连续目标 `E_5 - E_best`、`E_8 - E_best` 和成对 `E_low - E_high` 用于描述静态 schedule compute benefit；真正的因果跳算收益只由锁定 gate 的真实在线 run/skip 回放确认。

**备选方案：人工逐 chunk 标 easy/hard。** 不采用为主方案，因为成本高、标准主观，而且“人看起来复杂”不等于“模型多算有收益”。可以随机抽取约 100 个 clip 做盲审，只用于检查 phase proxy。

### 2. ground-truth 夹爪阶段只做解释性代理

从真值夹爪轨迹的稳定区间和状态切换构造：

- `free/open`：夹爪处于稳定开放模态，窗口内无切换；
- `pre-close`：首次向闭合模态切换前的短窗口；
- `closing/contact-proxy`：向闭合模态过渡的窗口；
- `hold`：闭合后保持稳定的窗口；
- `release`：从闭合模态向开放模态切换的窗口。

开放/闭合方向和阈值从数据双峰或 metadata 校准，不硬编码数值方向。由于数据没有接触力、物体位姿或距离，`contact-proxy` 只能表示“可能发生接触的阶段”。它用于回答静态 schedule compute benefit 是否在关键阶段升高，绝不用于定义 schedule 充分性，也不进入在线 gate。

### 3. 使用严格配对的 episode/chunk 实验

样本主键为 `episode_id + anchor_frame`。每个预算服务读取同一个冻结 sample manifest，确保任务文本、四帧观测、当前状态、24-step 真值动作、初始噪声 seed 完全一致。服务启动顺序和 warmup 不得改变样本对齐。

数据按 episode 划分 train/validation/test；同一 episode 的相邻窗口不得跨集合，以避免场景、语言和轨迹泄漏。首轮诊断集默认选择 50 个 episode、每个最多 10 个非重叠 chunk，按任务和夹爪阶段代理分层，约 500 个样本/配置。另从未参与阈值选择的 episode 构造保持自然阶段占比的锁定确认集；分层诊断集只用于发现反例和估计分组效应，最终总体收益以自然分布确认集为准。

WAM 服务具有跨 chunk 的 causal/KV-cache 状态。各静态 schedule 必须以相同顺序重放同一 episode 前缀，使用一致的初始单帧 priming、anchor、session 边界和 reset 时机。若采用随机 anchor 独立评估，必须为每个 schedule 重建同样的前缀或使用完全一致的隔离 reset+priming 协议；不能让一个服务继承历史而另一个服务从空 cache 开始。

### 4. 在训练归一化域中评价关节动作

服务输出是绝对关节目标，但模型训练的是相对动作。评估先统一表示：

```text
r_pred[t, j] = q_pred_abs[t, j] - q_current[j]
r_gt[t, j]   = q_gt_abs[t, j]   - q_current[j]
z_pred = q99_scale_without_clipping(r_pred)
z_gt   = q99_scale_without_clipping(r_gt)
```

主执行 horizon `H_exec` 必须在查看 test 结果前根据下游控制器实际消费的动作数写入 manifest，不能在不同策略间变化。当前 DROID pilot 使用 `H_exec=8`；正式报告同时提供 `H={1, 6, 8, 24}` 的敏感性结果。主指标为前 `H_exec` 个控制步、7 个关节的 `mean(abs(z_pred - z_gt))`。q99 只用于逐维缩放，计算误差前不得 clip，以免尾部错误被截断；同时报告未归一化的 raw-radian MAE 便于审计。

次指标包括：

- 完整 24-step 归一化 joint MAE；
- 第 `H_exec` 步和第 24 步 endpoint error；
- gripper MAE；
- 夹爪开/闭事件 F1、提前或滞后步数；
- 每策略实际 DiT 调用数、稳态端到端延迟和 gate 开销；
- 均值、P90、P95，以及 phase/task 分层结果。

关节和夹爪先分别报告，不用任意固定权重合成分数。由于示范数据具有动作多模态性，离线 GT 误差不是任务成功率的替代；如果离线结果通过，再将仿真或机器人任务成功率作为后续外部验证。

### 5. 分三类特征验证师兄的直觉

| 代号 | 只使用决策时可获得的信息 | 目的 |
|---|---|---|
| V | 相邻 video flow cosine 与现有 countdown | 复现当前 video-only 基线 |
| A | 有效 8 维 action flow 的 cosine、相对 L2、provisional action 变化 | 测试动作流是否比视频流更晚或更早收敛 |
| M | 相对位移、路径长度、速度、加速度/jerk、减速、方向变化、尺度无关 roughness、路径低效度、夹爪范围/最大步长/事件时刻 | 测试动作几何与夹爪事件是否能预测计算收益 |
| A+M | A 与 M 的可解释组合 | 测试收敛信号和物理动作信号是否互补 |

尺度无关 roughness 使用类似 `sum(norm(delta2_q)) / (sum(norm(delta_q)) + epsilon)` 的形式，避免大幅动作天然得到更高复杂度。action-flow cosine 必须配合范数保护和相对 L2；接近零向量时不单独使用 cosine。

第一版只比较阈值规则、逻辑回归和浅层决策树。复杂神经 gate 会增加过拟合风险、推理开销和解释难度，不适合在 H1/H2 尚未成立时引入。

### 6. 基线、oracle 与统计方法

必须同时报告：

- always-5、always-6、always-8、always-16；
- random gate，匹配候选策略的平均 DiT 调用；
- 当前 video-only 动态 gate；
- A、M、A+M 候选；
- 使用真值误差选择预算的 oracle，仅作为上界。

阈值/模型只在 train 拟合，在 validation 选择，test 一次性锁定评估。置信区间按 episode 做配对 bootstrap，不能把同一 episode 的 chunk 当作相互独立样本。比较 gate 时同时看 Pareto 前沿和默认验收阈值，避免只报“平均更快”而掩盖关键阶段或 P95 退化。

### 7. full-16 trace、prefix-stop 因果回放与真实动态 gate

完整 16-call 运行可保存每一步的 action/video flow 轻量统计，用来观察何时收敛、筛选特征和拟合 compute-benefit predictor。需要区分两种回放：

- **prefix-stop**：前 k 次与 full-16 完全一致，k 之后不再调用模型，只复用第 k 个 flow 到 scheduler 结束。因为停止后没有新的模型调用依赖分叉 latent，所以可由 full-16 的前 k 个 flow 精确重放。本实验验证 prefix-16 与正常 final 最大差为 0。
- **skip 后再 resume**：后续模型调用会读取已经分叉的 latent，不能用 full-16 后续 flow 替代，仍必须真实在线运行。

因此实验分两层：

1. **离线发现层**：利用配对固定预算和 full-16 prefix-stop 验证 H1/H2、选择少量 gate。
2. **在线确认层**：仅当 H2 go 时，把锁定 gate 接入真实 scheduler，在原 test sample manifest 上重新推理。

本轮 H2 no-go，因此第二层按预注册停止规则不执行。

### 8. 在线调度扩展现有 `should_run_model`

新增统一模式 `DIT_SCHEDULE_MODE=fixed|video|action|combined`，但兼容逻辑为：

- 未设置 `DIT_SCHEDULE_MODE` 时，完全沿用现有 `NUM_DIT_STEPS` 与 `DYNAMIC_CACHE_SCHEDULE` 解释；
- `fixed` 和 `video` 复用当前路径；
- `action` 使用 A 特征；
- `combined` 使用验证通过的 A+M 配置；
- gate 参数通过版本化 JSON 配置加载，并校验 checkpoint、embodiment、有效动作 mask 和特征版本。

调度器至少计算前两步。每一步 run 后保留最近两次 action/video flow 的轻量统计；skip 时复用最近一次完整预测，并照常执行两个 scheduler update。任何 NaN、零范数、缺失 schema、配置不兼容或异常都采用 fail-compute 策略，即运行 DiT 而不是跳过。

每个推理请求开始时只重置本次 16-step 去噪所用的 flow 历史、gate 特征和 countdown，防止上一个 chunk 的 gate 状态污染当前决策；WAM 的 causal 视频/KV-cache 仍按同一 session 连续保留。只有 `session_id` 变化或显式 episode reset 才清空 session 级 cache。trace 默认关闭；启用时只记录 step、timestep、特征标量、决策原因和累计调用数，不保存大型 flow tensor。

### 9. 明确的阶段门和停止条件

| 阶段结果 | 决策 |
|---|---|
| 静态 schedule 误差无稳定差异，或最低调用数可接受 schedule 几乎为常数 | H1 失败；停止 gate 实现 |
| H1 成立，但 V/A/M/A+M 均不能优于 random matched-compute | H2 失败；保留评估工具，不接在线 gate |
| A 有效，M 无效 | 继续 action convergence scheduler，但不使用“接触难度”叙事 |
| M 或 A+M 在 test 上预测 compute benefit | 支持动作特征与计算难度相关，再做在线回放 |
| 在线回放不满足均值、P95、关键阶段或 20% 节省阈值 | no-go；默认路径不变 |
| 在线回放完整通过 | 保留 opt-in gate，开始更大规模/任务成功率验证 |

本轮落入第二行：阶段可识别，但 V/A/M/A+M 和 GT phase oracle 均不能可靠预测 compute benefit。结论为 no-go，默认路径保持不变。

## Experiment Plan

### Phase 0：对齐、单元测试与 smoke test

1. 建立冻结 sample manifest、episode-level split 和 phase proxy。
2. 在 3-5 个 episode 上验证输入帧、当前状态、24-step 真值动作和服务输出严格对齐。
3. 验证 5/6/8/16 的实际 DiT 调用数、固定 seed 可复现、输出 `(24, 8)` 且有限。
4. 为 full-16 和 video-only 增加轻量 action/video flow trace。
5. 明确 warmup/编译样本不计入稳态延迟。

### Phase 1：配对 pilot，先回答 H1

1. 选择 50 个 episode、约 500 个非重叠 chunk，覆盖 free、pre-close、closing/contact-proxy、hold、release。
2. 同时启动四个双卡服务：
   - GPU 0-1：static-5；
   - GPU 2-3：static-6；
   - GPU 4-5：static-8；
   - GPU 6-7：static-16。
3. 四组完成后追加 video-only 动态基线；全部使用 cuDNN SDPA、相同 checkpoint 和 sample manifest。
4. 生成最低调用数可接受 schedule 分布、成对 compute-benefit 图、phase/task 分层表与 bootstrap CI。
5. 在保持自然阶段占比的锁定确认集上复核总体质量-计算量结论；分层诊断集不能替代该确认。
6. 若 H1 失败，立即停止，不为“难度判断”训练分类器。

按当前 static-16 稳态约 7.14 s/chunk 粗估，500 个样本的慢速组约 1 小时，加上模型加载、warmup、数据读取和失败重试，pilot 预留 1.5-2 小时墙钟时间。

### Phase 2：特征验证，回答 H2

1. 从决策时刻可获得的数据构造 V、A、M、A+M 特征。
2. 先画单特征与 `E_5 - E_best`、成对 compute benefit、最低调用数可接受 schedule 的关系，检查非线性和反直觉区间。
3. 训练阈值、逻辑回归和浅层树；在 validation 上锁定超参数。
4. 在 test 上报告静态 schedule 充分性标签的 balanced accuracy、macro-F1、AUROC 或有序分类指标，同时更重视最终 matched-compute Pareto。
5. 与 random、video-only 和 oracle 比较，并按决策表决定是否进入在线集成。

分类指标只用于诊断，最终目标不是“标签预测准确率最高”，而是在相同动作质量下减少 DiT 调用。

### Phase 3：真实在线 gate 回放，回答 H3

1. 只实现 Phase 2 中最多两个候选 gate，避免 test 上反复挑选。
2. 在冻结 test manifest 上运行真实 run/skip 轨迹。
3. 记录每步 decision trace、调用次数、gate 开销、端到端稳态延迟和全部动作指标。
4. 按默认 20%/2%/5% 阈值输出 go/no-go。
5. 通过后再扩展到至少 500 个 episode，直到主要差异的 episode-bootstrap CI 稳定。

### Phase 4：外部有效性

如果离线动作误差与在线计算验收均通过，再选择可用的仿真或机器人 replay 评估任务成功率、碰撞、接触阶段稳定性和闭环延迟。该阶段用于确认 GT action error 的代理有效性，不阻塞前面关于“是否值得实现 gate”的低成本判断。

## Result Artifacts

默认结果目录建议为 `outputs/action_aware_dit/<run_id>/`：

```text
manifest.json           # 代码、checkpoint、数据、硬件和推理配置
splits.json             # episode-level train/validation/test 与 sample keys
samples.jsonl           # 每样本、每预算的动作与指标
traces.jsonl            # 可选的逐 scheduler step 轻量特征与决策
summary.json            # 聚合指标、置信区间、schedule 充分性、compute benefit 和 go/no-go
report.md               # 人类可读的实验结论与限制
```

默认不复制原始视频，也不保存每一步大型 latent/flow；记录 episode/frame 引用即可。需要人工审查时，只导出少量指定 clip 或复用已有 `debug_image` 产物。

## Risks / Trade-offs

- **[GT 动作具有多模态性]** → 以首 6 步为主、分别报告 gripper 事件，并在通过后增加闭环任务成功率验证。
- **[夹爪阶段不等于真实接触]** → 统一命名为 contact-proxy，记录规则和样本量；需要语义结论时再做小规模人工或传感器标注。
- **[预算 mask 的步骤位置不同]** → 保存完整 step mask，不把 call count 当作唯一变量；最终必须用真实在线 gate 回放。
- **[单 seed 标签噪声]** → pilot 先固定 seed 保证成本可控；若 schedule 充分性标签大量落在 tolerance 边缘，对子集做多 seed 重复并使用连续 compute-benefit 目标。
- **[分层样本改变真实阶段占比]** → 分层集只用于诊断，另设锁定的自然分布确认集报告总体收益。
- **[跨 chunk cache 历史不一致]** → 按 episode 顺序重放并固定 priming/session/reset 协议；随机 anchor 必须重建同样前缀。
- **[full-16 trace 产生反事实偏差]** → trace 只用于候选发现，最终结论来自真实 skip 轨迹。
- **[action-flow 近零导致 cosine 不稳定]** → 同时使用范数保护、相对 L2 和 provisional action 变化，异常时 fail-compute。
- **[episode 内样本相关导致置信区间过窄]** → split 与 bootstrap 均以 episode 为单位。
- **[全成功示范数据的选择偏差]** → 结论限定为成功示范分布上的开放环动作预测，不外推到失败恢复和分布外状态。
- **[模型加载或编译污染延迟]** → 独立报告 cold-start 与 steady-state，配对比较只使用 warmup 后样本。
- **[新 gate 破坏现有推理]** → 新模式默认关闭，兼容测试覆盖旧环境变量，异常回退为运行 DiT。

## Migration Plan

1. 先加入只读评估和 trace instrumentation，不改变任何默认调度决策。
2. 完成 Phase 0/1 并保存可审计报告；H1 不通过则停止。
3. 完成 Phase 2，只有满足 H2 的候选才进入 scheduler 代码。
4. 以新 `DIT_SCHEDULE_MODE` 显式启用 action/combined gate，未设置时保持旧逻辑。
5. 完成真实在线回放和回归测试后，仍保持 action-aware 默认关闭。
6. 回滚时取消新模式或删除 gate 配置即可恢复原固定/video-only 路径；结果数据不影响模型或数据集。

## Open Questions

- 静态 schedule 充分性 tolerance 的三档具体值需要由 Phase 0 的重复推理噪声和 Phase 1 的误差分布校准，不能在看到 test 结果后修改。
- gripper 开/闭阈值应优先复用 checkpoint metadata；若 metadata 不足，则只在 train split 上拟合双峰阈值。
- 如果离线 GT error 与任务成功率相关性较弱，Phase 4 应选用哪个闭环环境或真实机器人 replay 指标，需要根据现有可用评估设施另行确定。
- 如果 M 特征不足以表达“爪离物体的距离”，后续可单独研究视觉 object-distance/contact estimator；这不应混入 v1 action-only 实验。
