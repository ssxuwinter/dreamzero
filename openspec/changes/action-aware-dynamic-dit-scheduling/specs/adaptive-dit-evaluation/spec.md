## ADDED Requirements

### Requirement: 配对预算扫描
评估系统 SHALL 对同一个 DROID 样本使用相同模型、相同输入观测、相同初始噪声和相同随机种子，分别执行固定 5、6、8、16 次 DiT 调用以及现有 video-only 动态调度。16 个 scheduler 积分步骤 SHALL 始终完整执行，固定 schedule 只改变实际调用 DiT 的步骤。系统 SHALL 将 5、6、8、16 记录为具有明确 step mask 的静态 schedule policy，不得仅凭调用次数把不嵌套的 mask 解释为同一路径上的因果计算预算。

#### Scenario: 同一样本完成全部预算推理
- **WHEN** 评估器收到一个合法的 episode 与 anchor frame
- **THEN** 系统为该样本生成 5、6、8、16 和 video-only 五组可按样本键配对的结果
- **THEN** 每组结果记录实际 DiT 调用次数，而不是仅记录配置名称

#### Scenario: 固定预算结果非单调
- **WHEN** 某样本的 16-call 误差高于 5、6 或 8-call 误差
- **THEN** 系统保留实测结果，并且不得假定更多 DiT 调用一定更优

#### Scenario: 比较调用数相同但步骤位置不同的 schedule
- **WHEN** 两个候选使用相同 DiT 调用数但 step mask 不同，或低调用与高调用 mask 不嵌套
- **THEN** 系统分别记录 policy id 和完整 step mask，并把误差差异解释为 schedule-policy 差异而不是纯调用次数效应

### Requirement: 可复现实验清单
每次实验 SHALL 保存足以重放该次评估的 manifest，包括代码版本与工作区状态、checkpoint 标识、数据集路径或版本、推理配置、DiT step mask、随机种子、attention backend、设备分配、输入帧偏移、anchor 规则和误差容差。

#### Scenario: 重新运行已有实验
- **WHEN** 使用某次实验保存的 manifest 对同一样本重新推理
- **THEN** 系统能够恢复该次实验的预算、输入对齐和随机性设置

### Requirement: 有状态推理与延迟协议
评估系统 SHALL 为各 schedule 固定 episode 顺序、初始单帧 priming、输入前缀、session 边界和 reset 时机。连续 chunk 评估 SHALL 保留同一 episode 内的 WAM causal/KV-cache；随机 anchor 评估 SHALL 为所有 schedule 重建相同前缀，或使用相同的隔离 reset+priming 协议。系统 SHALL 分开记录模型 cold start、首个多帧 warmup 请求和 warmup 后稳态延迟。

#### Scenario: 连续重放一个 episode
- **WHEN** 多个 anchor 来自同一 episode 并按时间顺序推理
- **THEN** 每个 schedule 使用独立但边界一致的 session，并继承相同长度和内容的 episode 前缀

#### Scenario: 报告稳态延迟
- **WHEN** 首个多帧请求包含一次性 scheduler、kernel 或缓存初始化开销
- **THEN** 系统单独报告该请求，且只用预先定义的后续 warm 请求统计 steady-state latency

### Requirement: 样本对齐与数据划分
评估系统 SHALL 以 episode 为最小划分单位建立 train、validation 和 test 集，并 SHALL 禁止同一 episode 或重叠动作窗口跨集合出现。首轮 pilot SHALL 同时包含按夹爪阶段代理与任务分层的诊断集，以及从未参与阈值选择的 episode 中按自然阶段占比抽取的锁定确认集。分层诊断集不得单独用于估计部署分布上的总体计算节省。

#### Scenario: 检测到 episode 泄漏
- **WHEN** 同一 episode 被分配到两个或更多数据划分
- **THEN** 评估器拒绝运行并报告冲突的 episode 标识

#### Scenario: 构造首轮 pilot
- **WHEN** 生成默认 pilot 样本清单
- **THEN** 系统为分层诊断集选择 50 个 episode、每个 episode 最多 10 个非重叠 anchor，并另行生成自然分布确认集，同时报告两套清单中每种夹爪阶段代理的样本数与采样权重

### Requirement: 结构化逐样本记录
评估结果 SHALL 至少记录 episode id、anchor frame、任务文本、当前关节状态、输入帧索引、session/cache 协议、ground-truth 动作、各 schedule 预测动作、`H={1, 6, 8, H_exec, 24}` 指标、gripper 指标、端点指标、实际 DiT 调用数、cold/warmup/稳态延迟、逐步 action/video 收敛特征、配置标识和失败状态。外部 DROID 预测动作 SHALL 保持 `(24, 8)`，且所有参与指标的值 SHALL 为有限值。

#### Scenario: 记录合法 DROID 预测
- **WHEN** 服务返回一个 `(24, 8)` 的有限动作数组
- **THEN** 评估器使用 7 个关节维度和 1 个夹爪维度分别计算指标并写入结果集

#### Scenario: 预测形状或数值非法
- **WHEN** 预测形状不是 `(24, 8)` 或包含 NaN/Inf
- **THEN** 评估器标记该样本失败、保留诊断信息，并且不得把它静默纳入聚合指标

### Requirement: 动作质量指标
关节误差 SHALL 在统一的相对关节动作表示中计算：预测和真值均相对当前关节状态表示，并使用 checkpoint 对应的逐关节 q99 训练统计进行缩放。主执行 horizon `H_exec` SHALL 在查看 test 结果前依据下游控制器实际消费的动作数写入 manifest；当前 DROID pilot 使用 8。主指标 SHALL 为前 `H_exec` 个控制步、7 个关节维度上的归一化绝对误差。q99 缩放后 SHALL 不在误差计算前 clip，并 SHALL 同时报告 raw-radian MAE；`H={1, 6, 8, 24}` 敏感性、端点误差、夹爪 MAE、夹爪事件 F1 与事件时序误差 SHALL 作为次指标单独报告。

#### Scenario: 计算主指标
- **WHEN** 一个样本同时具有当前状态、预测动作、真值动作和有效 q99 统计
- **THEN** 系统先将预测与真值转换到相同的相对动作域，执行不裁剪的逐关节 q99 缩放，再计算前 `H_exec` 步归一化误差和 raw-radian MAE

#### Scenario: 关节与夹爪结果汇总
- **WHEN** 聚合某个策略的结果
- **THEN** 系统分别报告关节指标和夹爪指标，不得用一个未经验证的固定权重把两者合成唯一分数

### Requirement: 静态 schedule 充分性与 compute-benefit 目标
系统 SHALL 对每个样本计算各静态 schedule 的主误差 `E_s`、样本内最优误差 `E_best = min(E_5, E_6, E_8, E_16)`、满足 `E_s <= E_best + tolerance` 的可接受 schedule 集合，以及其中实际调用数最少的 schedule。系统 SHALL 同时报告 `E_5-E_best`、`E_8-E_best` 和成对 `E_low-E_high` 连续 compute benefit。tolerance SHALL 可配置，并 SHALL 至少报告三档敏感性分析。最低调用数可接受 schedule SHALL 被明确标记为静态 policy 标签，不得称为同一轨迹上的因果 required budget。

#### Scenario: 低预算已在容差内
- **WHEN** 5-call 误差不超过该样本最优误差加 tolerance
- **THEN** 5-call 被加入可接受 schedule 集合；若其调用数最少，则最低调用数可接受静态 schedule 标记为 5

#### Scenario: 只有高预算满足容差
- **WHEN** 5、6、8-call 均超出容差而 16-call 达到样本内最优误差
- **THEN** 最低调用数可接受静态 schedule 标记为 16，但报告仍注明该结论同时受 16-call step mask 位置影响

### Requirement: 阶段代理仅用于解释
系统 SHALL 从 ground-truth 夹爪轨迹构造 free/open、pre-close、closing/contact-proxy、hold 和 release 等阶段代理，并 SHALL 明确这些标签不是传感器级真实接触标签。阶段代理 SHALL 只用于分层分析，不得作为静态 schedule 充分性或 compute benefit 的定义，也不得作为在线 gate 的输入。动作特征能够区分阶段本身，不构成增加计算量的充分证据。

#### Scenario: 分析接触代理子集
- **WHEN** 报告 closing/contact-proxy 或 hold 子集的结果
- **THEN** 报告中同时给出代理规则和样本量，并注明未使用物体距离、接触力或人工接触标签

### Requirement: 基线与候选策略比较
评估 SHALL 比较 always-5、always-6、always-8、always-16、同平均计算量 random gate、现有 video-only gate、action-flow 收敛 gate、动作/夹爪特征 gate、组合 gate 和使用真值误差选择最低调用数可接受静态 schedule 的 oracle。任何可学习阈值或浅层模型 SHALL 只在 train 集拟合、在 validation 集定参，并仅在锁定后评估 test 集。

#### Scenario: 与随机和 video-only 基线公平比较
- **WHEN** 比较 action-aware 候选策略
- **THEN** 系统在相同或经插值匹配的平均 DiT 调用数下报告其误差，并同时展示 random gate 与 video-only gate

#### Scenario: 展示理论上界
- **WHEN** 已获得每个 test 样本的全部固定预算结果
- **THEN** 系统报告静态 schedule oracle 的质量-计算量点，但不得将 oracle 使用的真值信息提供给在线 gate，也不得把非嵌套静态 oracle 当作真实在线 gate 可达到的上界

### Requirement: 统计报告与 go/no-go 判定
系统 SHALL 按 episode 进行配对 bootstrap，报告均值、P90、P95 和置信区间，并输出机器可读的 go/no-go 结论。默认 go 标准 SHALL 要求：相对当前默认 always-8，平均实际 DiT 调用至少减少 20%；主指标均值退化不超过 2%；P95 和 contact-proxy 子集退化均不超过 5%；并且候选策略在匹配计算量下 Pareto 优于 random gate 与现有 video-only gate。

#### Scenario: 计算收益不存在异质性
- **WHEN** 高调用 schedule 没有改善达到最小样本占比的任何稳定子集，或几乎所有样本的最低调用数可接受 schedule 相同
- **THEN** 系统输出 no-go，并停止把动作信号用于样本级计算分配的后续实现

#### Scenario: 仅收敛信号有效
- **WHEN** action-flow 收敛信号预测 compute benefit，但动作/夹爪物理特征不能预测 compute benefit
- **THEN** 系统可以对自适应收敛调度输出 go，但 SHALL 明确拒绝“接触语义难度已被验证”的结论

#### Scenario: 满足完整验收标准
- **WHEN** action-aware 策略满足默认质量、尾部误差、关键阶段和计算节省阈值
- **THEN** 系统输出 go，并保存支持该结论的 test 结果、置信区间和所用阈值

### Requirement: 可恢复的多 GPU 执行
预算扫描 SHALL 支持相互独立的服务并行执行和按样本键断点续跑。默认八卡拓扑 SHALL 允许四个双卡服务分别运行 5、6、8、16-call 配置，video-only 基线可在固定预算扫描后追加运行。

#### Scenario: 某个预算服务中断
- **WHEN** 一个服务在完成部分样本后退出
- **THEN** 重启后系统跳过已通过完整性校验的样本，并继续缺失样本而不覆盖其他预算结果

### Requirement: 同轨迹 prefix-stop 因果评估
评估系统 SHALL 能在默认关闭的研究模式下记录 full-16 的有效 8 维 action flow，并对每个 `k=1..16` 重放 prefix-stop：前 k 个 flow 与 full-16 完全一致，之后不再调用 DiT，复用第 k 个 flow 完成剩余 scheduler 积分。系统 SHALL 验证 prefix-16 与正常 final 数值一致，并 SHALL 将该结果解释为“停止后不再 resume”的因果前缀策略；不得把它外推到 skip 后仍会 resume 模型调用的动态轨迹。

#### Scenario: 验证 prefix 回放对齐
- **WHEN** 对一个 full-16 样本生成 `k=1..16` prefix-stop 动作
- **THEN** prefix-16 与正常 final 的最大绝对差不超过预先声明的数值容差，并记录共享 seed、初始 action noise 和 scheduler 配置

#### Scenario: prefix-stop 与动态 skip-resume 的边界
- **WHEN** 候选策略在 skip 后还会执行新的 DiT 调用
- **THEN** 系统拒绝使用 full-16 后续 flow 代替真实分叉轨迹，并要求实际在线回放

### Requirement: 阶段信号与计算收益必须分别判定
评估系统 SHALL 分别报告 phase prediction 与 compute-benefit prediction。即使动作/夹爪特征的 phase AUROC 很高，只要 episode-held-out compute-benefit 预测未过预注册门槛，系统 SHALL 输出 phase-gate no-go，并停止把“精细阶段”用作增加 DiT 计算的依据。

#### Scenario: 阶段可识别但计算收益不可预测
- **WHEN** 动作特征可以区分 free/fine，但动作特征、action flow 和 GT phase oracle 均不能可靠预测额外计算收益
- **THEN** 系统保留阶段分析结果，拒绝实现 phase-aware more-compute gate，并保持默认推理路径不变
