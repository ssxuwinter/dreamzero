# H1：GT 动作 chunk 间阶段代理实验协议

状态：锁定后执行。

## 目的

用大样本 DROID 示范检验动作几何在“自由移动/接近闭合/闭合/保持/释放”之间是否存在稳定差异，并验证 chunk 间特征是否相对 chunk 内特征有增量信息。该实验只回答阶段可识别性，不回答 WAM 是否值得多算。

## 数据与采样

- 固定随机种子 `20260719`。
- 从成功 episode 中等距/随机结合抽取 1,200 个 episode；episode 是分组和 bootstrap 单位。
- 每 24 帧形成一个非重叠输出 chunk，anchor 从 23 开始；不足 24 个未来动作的尾部丢弃。
- 特征只使用当前 chunk 及前一个 chunk；阶段标签可使用当前/未来 24 步 GT 夹爪，仅用于离线评估。

## 阶段代理

根据每个 episode 内归一化后的 GT gripper 轨迹和迟滞阈值构造：

- `free_open`：当前开放，当前 chunk 无闭合/释放事件，且至少距离首次闭合两个 chunk。
- `pre_close`：开放，当前 chunk 无闭合，但下一 chunk 出现闭合事件。
- `closing`：当前 chunk 出现开放到闭合事件。
- `hold`：当前闭合且当前 chunk 无释放事件。
- `release`：当前 chunk 出现闭合到开放事件。
- 阈值附近或 episode 无清晰开/闭两态的样本标记 `ambiguous`，不进入主指标。

该标签是夹爪阶段代理，不是真实接触或物体距离标签。

## 特征组

- `joint_within`：路径长度、平均/最大 step、endpoint displacement、加速度、jerk、方向变化、roughness。
- `gripper_within`：范围、total variation、最大 step、事件计数、起点/终点状态。
- `transition`：相邻 chunk 边界不连续、上述统计量的一阶差、endpoint 方向变化、夹爪状态/事件变化。
- 主比较：`joint-only`、`gripper-only`、`joint+gripper`、`joint+gripper+transition`。

## 指标与防泄漏

- 主任务：`fine = pre_close|closing|hold|release` 对 `free_open`。
- 次任务：五分类 macro-F1；重点报告 `pre_close` recall。
- 使用 episode-grouped 5-fold cross-validation；所有标准化、缺失填充和模型拟合仅在训练 fold 内完成。
- 使用 Logistic Regression 作为主模型，Random Forest 作为非线性敏感性分析。
- 报告 AUROC、AUPRC、balanced accuracy、macro-F1 和 episode bootstrap 95% CI。
- 随机打乱标签作为 sanity baseline；不允许按 chunk 随机切分。

## 判定规则

- H1 支持：`joint+gripper+transition` 在 grouped CV 中稳定优于 `gripper-only`，且对 `pre_close` 有可复现增益。
- H2 支持：joint 幅度/速度/jerk 在各阶段的效应方向不稳定，或 `joint-only` 明显弱于含夹爪事件的模型。
- 即使 H1 支持，也不得推出“fine 阶段需要更多 DiT”。

