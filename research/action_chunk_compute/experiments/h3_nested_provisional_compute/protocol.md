# H3：同一去噪轨迹的 provisional-action 动态计算实验协议

状态：待 H2 完成后执行。

## 目的

严格测量“当前推理已经算到某一步时，继续调用 DiT 是否有价值”。这也是师兄设想真正需要的在线证据。

## 核心设计

- 在完整 16-step scheduler 轨迹中记录每个实际 DiT 调用后的 provisional normalized action、action flow 和 video flow 摘要。
- 主因果回放固定为 prefix-stop：对 `k=1..16`，前 `k` 个 action flow 与 full-16 完全相同，之后不再调用 DiT，而是复用第 `k` 个 flow 完成剩余 scheduler 积分。
- prefix-stop 输出由一次 full-16 trace 离线重放得到；初始 action noise、前 `k` 次模型状态和 scheduler 配置完全配对。`k=16` 必须与服务正常 final action 数值一致。
- 每个候选停止点 `s` 的真实收益定义为：从同一个 `x_s` 继续运行到完整轨迹后，动作误差相对在 `s` 停止/复用 flow 的改善。
- 所有 gate 特征必须在停止点 `s` 已经可用；不得使用最终动作、未来 flow、GT phase 或未来图像。
- 记录 action update norm、相邻 provisional action cosine/L1、预测夹爪事件稳定性、action-flow cosine、video-flow cosine，以及前一最终 chunk 的 transition 特征。
- 主检查点为 `k={2,4,8}`，其余 `k` 用于绘制完整 quality-compute 曲线；收益阈值沿用 H2 的 `{0, 0.002, 0.005}`。

## 比较策略

- 固定 call 数。
- 当前 video-only dynamic cache。
- action-flow-only。
- action provisional convergence-only。
- video+action 联合 gate。
- phase/previous-chunk 特征只作为增量项，不单独决定当前 chunk。
- 与各策略平均调用量匹配的 random gate。

## 主指标

- 平均实际 DiT call 数与稳态延迟。
- H8 q99-normalized joint MAE、gripper MAE。
- 全体、P95、pre-close/contact-proxy 子集退化。
- quality-compute Pareto 曲线和 paired episode bootstrap 95% CI。

## 上线门槛

相对默认 8-call：平均 DiT 调用减少至少 20%，H8 主误差均值退化不超过 2%，P95 与关键阶段退化不超过 5%；并且联合 gate 在匹配调用量下优于 random 与 video-only。否则结论是“不应按当前动作信号动态多算”。

## 时间因果约束

前一个最终 chunk 的输出可以调度下一个 chunk，但无法节省当前 chunk；当前 chunk 的早停必须依靠 provisional action/action-flow。最终报告必须分别呈现 lagged 调度和 same-request 调度，禁止混写。
