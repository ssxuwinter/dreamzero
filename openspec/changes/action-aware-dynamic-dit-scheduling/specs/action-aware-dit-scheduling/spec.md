## ADDED Requirements

> **2026-07-19 research status:** 前置 H2 go 条件未成立。本 delta 记录条件式设计，但本轮 SHALL NOT 应用 action-aware online scheduler；默认 fixed/video-only 路径保持不变。重新启用实现必须先提供新的闭环 compute-benefit 证据并更新评估结论。

### Requirement: 可选且向后兼容的调度模式
系统 SHALL 将 action-aware 调度作为显式选择的推理模式提供。未设置新配置时，现有固定预算默认值、`NUM_DIT_STEPS` 行为、`DYNAMIC_CACHE_SCHEDULE` 行为和外部动作输出合同 SHALL 保持不变。

#### Scenario: 未启用 action-aware 模式
- **WHEN** 用户沿用现有推理命令且不提供 action-aware 配置
- **THEN** 系统执行与变更前相同的固定预算或 video-only 调度路径

#### Scenario: 显式启用 action-aware 模式
- **WHEN** 用户选择一个已经通过离线验收的 action-aware 策略及其配置
- **THEN** scheduler 使用该策略决定后续 DiT 调用，并记录策略版本

### Requirement: 只使用有效动作维度
action-aware 特征 SHALL 根据 embodiment 的动作 schema 应用有效维度 mask。对 DROID，系统 SHALL 只使用 7 个关节维度和 1 个夹爪维度，并 SHALL 排除内部 32 维 action register 中的 padding 维度。关节特征与夹爪特征 SHALL 分别缩放和汇总。

#### Scenario: DROID 内部动作 flow 为 32 维
- **WHEN** scheduler 从内部 action-flow 张量提取 gate 特征
- **THEN** 只有与外部 `(24, 8)` 动作合同对应的维度参与相似度、距离和复杂度计算

#### Scenario: 缺少有效维度 schema
- **WHEN** action-aware 模式遇到无法确定有效动作维度的 embodiment
- **THEN** 系统禁用跳算并回退到配置的安全固定预算，同时输出诊断信息

### Requirement: action-aware 特征定义
候选 gate SHALL 支持 action-flow cosine similarity、相对 L2 变化和 provisional action 变化；动作/夹爪候选特征 SHALL 支持相对位移、路径长度、速度、加速度或 jerk、减速、方向变化、路径低效度、夹爪变化范围、最大单步变化和预计事件时刻。系统 SHALL 不得把原始绝对关节角大小单独解释为动作难度。

#### Scenario: action flow 已趋于稳定
- **WHEN** 连续有效 action-flow 的 cosine、相对 L2 和 provisional action 变化均达到已校准的稳定条件
- **THEN** gate 可以在满足最小计算量和安全条件时跳过后续一次或有限次数的 DiT 调用

#### Scenario: 夹爪事件或动作方向快速变化
- **WHEN** provisional action 显示夹爪切换、明显减速、方向变化或高尺度无关 roughness
- **THEN** 系统记录相应阶段候选特征；只有该特征已经在锁定 test 上证明能够预测 compute benefit、且所属 gate 已通过 matched-compute 验收时，组合 gate 才可将其用于 run/skip 决策

#### Scenario: 特征只能区分物理阶段
- **WHEN** 某动作或夹爪特征能够区分 free 与 contact-proxy，但不能预测额外 DiT 调用带来的质量收益
- **THEN** 系统只将该特征用于解释性报告，不得据此提高计算量或宣称 action-aware 调度有效

### Requirement: 单次在线去噪调度
action-aware scheduler SHALL 在一次 16-step scheduler 轨迹内作出 run/skip 决策，不得先完成低预算推理再从头执行高预算推理。无论是否跳过 DiT，video 与 action scheduler update SHALL 每一步都执行；跳过时 SHALL 复用最近一次有效的 video flow 和 action flow。

#### Scenario: gate 决定跳过当前 DiT 调用
- **WHEN** 当前步骤满足跳算条件且已有有效历史预测
- **THEN** 系统不调用 DiT，但仍用最近一次有效的 video/action flow 完成当前 scheduler update

#### Scenario: 开始新的推理请求
- **WHEN** 同一 session 内的新 chunk 或新推理请求开始
- **THEN** 系统清空前一个请求的逐步 flow 历史、skip countdown 和 gate 状态，但保留 WAM session 级 causal 视频/KV-cache

#### Scenario: 开始新的 episode
- **WHEN** `session_id` 变化或客户端执行显式 reset
- **THEN** 系统同时清空请求级 gate 状态和 session 级 causal/KV-cache，且该边界行为被 trace 或运行清单记录

### Requirement: 保守运行与故障回退
调度器 SHALL 至少执行最初两次 DiT 调用以建立历史，并 SHALL 提供最小调用数、最大连续跳过数和最大总调用数配置。若特征包含 NaN/Inf、flow 范数退化、历史不足、策略配置不兼容或 gate 执行失败，系统 SHALL 选择运行 DiT，而不是继续跳算。

#### Scenario: action-flow cosine 不可定义
- **WHEN** 连续 action-flow 至少一个范数小于数值稳定阈值
- **THEN** 系统不依据 cosine 触发跳算，并使用其他有效特征或保守地运行 DiT

#### Scenario: gate 运行异常
- **WHEN** gate 在某一步抛出异常或输出非法决策
- **THEN** 当前及后续步骤回退到安全策略，推理请求仍返回符合合同的结果并记录异常

### Requirement: 可解释且可版本化的 gate
首个 action-aware gate SHALL 使用阈值规则、逻辑回归或浅层决策树等可解释方法，不得引入新的大型神经网络。gate 配置 SHALL 包含特征列表、归一化统计、阈值或模型参数、训练数据 manifest 和版本标识。

#### Scenario: 加载已校准 gate
- **WHEN** action-aware 模式启动
- **THEN** 系统验证 gate 配置与 checkpoint、动作 schema 和特征版本兼容后才允许跳算

### Requirement: 逐步调度可观测性
系统 SHALL 为每个 scheduler step 提供可选 trace，至少包含 step index、timestep、run/skip 决策、决策原因、action/video 收敛分数、skip countdown 和累计 DiT 调用数。关闭 trace 时 SHALL 不保存大体积中间张量。

#### Scenario: 调试一次 action-aware 推理
- **WHEN** 用户启用 gate trace
- **THEN** 每一步均产生足以重建调度决策的轻量记录，且外部动作预测保持不变

### Requirement: 输出与确定性合同
action-aware 模式 SHALL 继续返回与现有 WAM 服务兼容的 `(24, 8)` 有限动作序列，并维持相同的关节绝对化与夹爪后处理语义。给定相同模型、输入、seed、gate 配置和确定性运行环境，调度决策 SHALL 可复现。

#### Scenario: 客户端使用 action-aware 服务
- **WHEN** 现有客户端向启用 action-aware 模式的服务发送合法请求
- **THEN** 客户端无需修改动作解析逻辑即可消费返回结果

### Requirement: 在线候选必须经过实际回放
系统 SHALL 使用真实 run/skip 调度重新运行锁定的 test 样本；仅根据 full-16 轨迹离线模拟出的结果不得作为最终性能结论，因为跳算会改变后续去噪状态和特征轨迹。

#### Scenario: 离线 trace 选择出候选 gate
- **WHEN** 候选 gate 在保存的 full-16 trace 上满足离线指标
- **THEN** 系统仍需以该 gate 完成实际在线回放并重新计算质量、尾部误差、延迟和调用次数
