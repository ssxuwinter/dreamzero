# H2：WAM 静态 schedule 配对实验协议

状态：锁定后执行；该实验是 compute proxy，因 schedule mask 不嵌套而保持探索性解释。

## 目的

在同一输入、同一 episode、同一 anchor 上比较 static-5、static-8、static-16 的 WAM 输出，回答：

1. WAM 预测动作能否复现 GT 研究中的阶段信号？
2. 最终动作 chunk 的 within/transition 特征能否预测某个样本在哪种静态 schedule 下误差更低？

## 样本

- 从 H1 数据中按 episode 选择至少 12 个 episode，覆盖清晰的 free、pre-close、closing、hold、release。
- 确认集 episode 与 pilot episode 0 分离。
- 每个 episode 使用连续非重叠 anchor，并为三个 schedule 使用相同帧、状态、prompt 和独立但等价的 session 顺序。
- 先运行小规模 3-episode engineering set 验证 runner，再锁定 manifest 执行确认集。

## 推理配置

- 三个双卡服务并行：5-call、8-call、16-call。
- T5 常驻 CPU 并按层临时上卡；`ATTENTION_BACKEND=cudnn`；encoder compile 关闭。
- 保存 git HEAD、dirty diff 摘要、checkpoint 路径、环境版本、step mask、每次延迟和实际 DiT call 数。
- 不更改模型随机种子、输入或后处理；若当前服务无法固定随机性，至少运行重复子集评估方差。

## 标签与指标

- 主误差：q99 range 归一化、未裁剪的 joint MAE H8；gripper H8 单独报告。
- 次误差：H24 joint/gripper、P95、阶段分组误差。
- 允许非单调：`E_best = min(E5,E8,E16)`；对每个 tolerance 求最低可接受 call 数。
- 探索性 compute proxy：`benefit_5_to_16 = E5 - E16`、`benefit_8_to_16 = E8 - E16`。
- 特征只来自低预算或前一 chunk 的 WAM 输出，不能使用高预算输出构造 gate 输入。
- 评估 episode-held-out 的 Spearman、AUROC（benefit > tolerance）与简单回归；报告 episode bootstrap CI。

## 基线

- 固定 5/8/16-call。
- 与目标预算匹配的随机 gate。
- GT phase-only oracle（仅作上限诊断，不可部署）。
- predicted-gripper-only、joint-only、joint+predicted-gripper、加 transition。

## 判定规则

- 若更多静态调用没有形成稳定收益子集，则 H3 得到支持，停止把物理阶段当作多算依据。
- 若存在收益异质性但动作特征无法 held-out 预测，则不实现 action-aware gate。
- 即使静态 proxy 可预测，也必须进入 H3 nested 实验才能作当前请求动态早停结论。

