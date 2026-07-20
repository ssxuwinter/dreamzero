# 研究日志

## 2026-07-19：研究初始化

- 将师兄的直觉拆成两个独立命题：动作输出能否识别精细操作阶段；该阶段或信号能否预测继续执行 DiT 的收益。
- 锁定三个层次的实验：GT 动作大样本阶段分析、WAM 静态 5/8/16 配对推理、单条 nested 去噪轨迹的 provisional-action 早停实验。
- 明确最终 chunk 输出只能用于下一 chunk 的滞后决策；若要决定当前 chunk 是否继续多算，必须使用当前推理中已经产生的 action-flow 或 provisional action。
- 已有单 episode pilot 支持“输出动作含有阶段信号”，但不支持“交互阶段更值得多算”。本工作区从多 episode 证据重新检验。
- 当前仓库含用户未提交改动，因此所有模型结果必须记录 git HEAD、dirty 状态、环境变量和 manifest；在锁定协议后才运行确认性实验。

## 决策记录

| 日期 | 决策 | 原因 |
|---|---|---|
| 2026-07-19 | 夹爪 GT 只构造阶段代理，不作为在线 gate 输入 | 避免未来信息泄漏与循环定义 |
| 2026-07-19 | joint-only、predicted-gripper-only、joint+predicted-gripper+transition 分组比较 | 分辨阶段信息究竟来自夹爪事件还是关节几何 |
| 2026-07-19 | 将静态 5/8/16 结果标记为探索性 compute proxy | 三个 step mask 不嵌套，不能解释为纯粹“多算几步”的因果收益 |
| 2026-07-19 | 最终判断以 nested 当前轨迹为准 | 只有从同一个早期状态继续运行才能测量真正的当前请求额外计算收益 |

## 2026-07-19：H1 大样本 GT 阶段实验

- 按冻结随机种子抽取 1,200 个成功 episode，得到 11,030 个连续 24-step chunk。
- 用 episode 内夹爪迟滞阈值构造 free/open、pre-close、closing、hold、release；356 个不清晰 chunk 标记 ambiguous。
- episode-grouped 5-fold 结果表明动作几何和 chunk transition 能提前区分 pre-close；full 特征 AUROC 0.800。
- closing 相比 free 更慢、更平滑，支持“大小/变化率不是正单调难度指标”。

## 2026-07-19：H2 WAM 静态 schedule 实验

- 冻结 12 条阶段完整的连续轨迹，每条 7 个 chunk；三组双卡服务并行运行 static-5/8/16。
- 三个 schedule 的重复子集逐元素差为 0；cuDNN SDPA、T5 CPU offload 和双卡推理均正常。
- 5/8/16-call H8 joint MAE 分别为 0.02108/0.02121/0.02174；16-call 没有总体收益。
- predicted action 能识别阶段，但所有动作特征和 GT phase 均不能 held-out 预测静态 compute benefit。

## 2026-07-19：H3 同轨迹 prefix-stop 实验

- 新增默认关闭的研究 trace；在 full-16 中保存 8 个有效动作维的 flow，并离线重放 `k=1..16` prefix-stop。
- prefix-16 与正常 final 及独立 static-16 逐元素一致，证明回放对齐。
- 84 个 chunk 上，离线 H8 joint MAE 从 1/2-call 的约 0.0182 上升到 16-call 的 0.02174。
- pre-close 的 8→16 benefit 约 0.00003，其余阶段为负；没有“精细阶段更需要多算”的证据。
- compute-benefit predictor 全部未过门槛，研究按预注册停止规则判为 no-go，不继续实现在线 action-aware scheduler。

## 最终决策

| 子命题 | 结论 | 后续 |
|---|---|---|
| 动作序列能否分出自由/精细阶段 | 支持 | 可作为轨迹分析或执行 chunk 自适应的独立课题 |
| 参数越大、变化越快是否越难 | 否定 | 使用减速、夹爪事件和 transition 的联合描述，不用单阈值 |
| 精细阶段是否值得更多 DiT | 否定于当前离线指标 | 不实现 phase-aware more-compute gate |
| action-flow 是否能预测继续计算收益 | 当前不支持 | 闭环验证前不实现 gate |
| 固定低预算是否可能更合适 | 值得闭环验证 | 比较 2/4/8-call 成功率、稳定性和安全性 |

