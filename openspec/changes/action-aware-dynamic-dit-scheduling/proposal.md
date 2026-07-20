## Why

DreamZero 当前只支持固定 DiT 计算预算，或仅依据视频 flow 收敛程度进行动态跳算；这两种方式都没有验证 WAM 输出的动作序列能否识别“需要更多计算”的样本。需要先完成一套可复现、可证伪的实验，把“接近接触时动作更难”与“增加 DiT 计算确实能改善动作预测”分别验证，避免仅凭动作幅度作出假设。

## Research Decision (2026-07-19)

本 change 已完成阶段富集诊断实验，并触发预注册 no-go 分支：

- 1,200 episode 的 GT 动作实验支持“动作可分出 free/pre-close/closing/hold/release”；`pre-close vs free` 的 joint+gripper+transition AUROC 为 0.800。
- 12 episode、84 个连续 WAM chunk 上，predicted-gripper 的 fine/free AUROC 为 0.911，证明输出动作含有阶段信号。
- static-5/8/16 的 H8 joint MAE 为 0.02108/0.02121/0.02174；高调用没有总体改善。
- 同一 full-16 轨迹的 prefix-stop 因果回放中，1/2/4/8/16-call H8 joint MAE 为 0.01818/0.01817/0.01946/0.02119/0.02174；所有阶段均未显示稳定的“精细时多算”收益。
- action-flow、provisional convergence、动作几何、chunk transition、lagged previous chunk 和 GT phase oracle 均未获得可接受的 held-out compute-benefit 预测力。

因此本 change 在当前证据下只保留评估、trace 和 no-go 结论，**不进入 action-aware online scheduler 实现**。若后续闭环成功率证明某些状态确实从额外计算受益，应以新证据重新提出或更新该条件式 capability。

## What Changes

- 新增配对离线评估：对完全相同的 DROID 观测分别运行固定 5、6、8、16 次 DiT 调用，并运行现有 video-only 动态调度基线。
- 以机器可读格式记录对齐后的动作预测、归一化动作误差、延迟、实际 DiT 调用次数、action/video 收敛轨迹，以及辅助性的夹爪阶段代理标签。
- 根据每个样本相对于样本内最优静态 schedule 的实测误差，在可配置容差下生成“最低调用数可接受静态 schedule”标签，并计算连续 compute benefit。由于 5、6、8、16-call 的 step mask 不嵌套，该标签只描述候选静态策略，不被解释为因果 required budget。
- 在质量-计算量 Pareto 曲线上比较固定预算、同平均计算量随机 gate、video-only、action-flow 收敛、动作/夹爪特征、组合 gate 与 oracle gate。
- 仅在预先约定的 go 条件成立后新增可选的 action-aware 动态调度；本轮条件未成立，因此停止该实现，仓库默认推理行为与输出合同保持不变。
- 明确 stop 条件：如果额外计算不存在有用的异质收益，或动作信号无法优于同计算量基线，则停止在线 gate 的实现或重新定义研究问题。
- 将分层诊断样本与保持自然阶段占比的锁定确认样本分开，并固定 episode/session/KV-cache、执行 horizon、归一化和稳态延迟协议，防止阶段重采样或缓存历史污染结论。

## Capabilities

### New Capabilities

- `adaptive-dit-evaluation`：为 DreamZero 动作推理提供可复现的配对预算扫描、计算收益目标构造、动作阶段分析与统计验收标准。
- `action-aware-dit-scheduling`：条件式在线 DiT 调度设计；本轮被 no-go 门阻止，不作为待实现 capability 应用到默认推理。

### Modified Capabilities

无。

## Impact

- 影响 `groot/vla/model/dreamzero/action_head/wan_flow_matching_action_tf.py` 中的 DreamZero WAM 推理路径和动态 cache scheduler。
- 新增 DROID episode/chunk 推理的评估工具与结果产物，不修改已下载的数据集或模型权重。
- 继续使用现有 `(24, 8)` DROID 动作输出合同；构造 action-aware 特征时必须排除模型内部 padding 的动作维度。
- 不需要新增模型权重，也没有破坏性 API 变更；action-aware scheduler 默认关闭。
- 完整配对扫描可在 8 张 RTX PRO 5000 上启动 4 个相互独立的双卡服务；该离线实验不依赖 sequence parallelism。
- 研究用 `TRACE_ACTION_DENOISING=true` 路径默认关闭，仅用于生成同轨迹 prefix-stop artifact；no-go 后不把大型 trace 或实验响应合同合入默认服务行为。
