## 0. 本轮研究结果与停止决定

- [x] 0.1 完成 1,200 episode GT chunk 阶段实验，验证阶段信号、pre-close transition 增量和“大/快等于难”的反例
- [x] 0.2 完成 12 episode x 7 chunk 的 static-5/8/16 严格配对、固定 seed 重复检查、episode bootstrap 和 compute-benefit held-out 分析
- [x] 0.3 完成同一 full-16 轨迹的 `k=1..16` prefix-stop 因果回放，并验证 prefix-16 与正常 final 逐元素一致
- [x] 0.4 输出 H2 no-go：阶段、动作几何、chunk transition、action flow、provisional convergence 和 GT phase oracle 均不能可靠预测额外计算收益
- [x] 0.5 按预注册停止规则取消本轮条件式 action-aware scheduler 和真实 adaptive gate 集成，保持默认推理路径不变
- [x] 0.6 生成 `docs/2_action_phase_compute_research.md` 与 `research/action_chunk_compute/` 中的协议、manifest、逐样本结果和机器可读 summary

> 下列原始任务保留为完整产品化路线。第 6、7 节因 0.4 的 H2 no-go 而不适用于本轮，不应误读为仍需继续实现在线 gate。

## 1. 实验数据与结果基础设施

- [ ] 1.1 在 `eval_utils/action_difficulty/` 建立评估模块，定义 run manifest、sample key、逐样本结果、逐 step trace 和 summary 的版本化 schema
- [ ] 1.2 实现 DROID episode/chunk 读取与严格对齐，输出任务文本、`[-23, -16, -8, 0]` 输入帧、当前状态和 24-step ground-truth 动作，并为边界样本添加校验
- [ ] 1.3 实现基于 ground-truth 夹爪轨迹的 free/open、pre-close、closing/contact-proxy、hold、release 代理生成器，记录阈值来源和不确定样本
- [ ] 1.4 实现 episode-level train/validation/test 划分与非重叠 anchor 抽样，检测并拒绝 episode 或窗口泄漏
- [ ] 1.5 实现 JSONL 结果写入、完整性校验、失败记录和按 `episode_id + anchor_frame + policy` 断点续跑
- [ ] 1.6 为 schema、样本对齐、阶段代理、split 泄漏和断点续跑添加 CPU 单元测试

## 2. 推理记录与动作指标

- [ ] 2.1 为 WAM 推理增加默认关闭的轻量 trace，记录 step index、timestep、run/skip、实际 DiT 调用数和现有 video-flow 收敛分数
- [ ] 2.2 在实际 DiT 调用后记录有效 8 维 action-flow 的 cosine、范数、相对 L2 与 provisional action 变化，验证 32 维 padding 不参与特征
- [ ] 2.3 保存每次运行的固定 step mask、seed、checkpoint、attention backend、设备、代码版本和 dirty-worktree 状态，且不默认保存大型 latent、flow 或预测视频
- [ ] 2.4 实现相对关节动作 canonicalization 和不裁剪的 checkpoint q99 缩放；从 manifest 读取 `H_exec`，计算 `H={1,6,8,H_exec,24}` 的归一化及 raw-radian 误差和对应端点误差
- [ ] 2.5 实现独立的 gripper MAE、开闭事件 F1、事件提前/滞后指标，以及 mean/P90/P95 聚合
- [ ] 2.6 添加回归测试，确认 trace 关闭时默认 8-call、`NUM_DIT_STEPS`、`DYNAMIC_CACHE_SCHEDULE` 和外部 `(24, 8)` 输出行为不变

## 3. Phase 0 对齐与 Smoke Test

- [ ] 3.1 生成覆盖至少 3-5 个 episode 的 smoke sample manifest，并人工核对输入帧、当前状态和 ground-truth 24-step 动作的时间对齐
- [ ] 3.2 对同一 smoke manifest 分别运行 static-5、static-6、static-8、static-16 和 video-only，验证实际 DiT 调用数、有限输出与样本键完整配对
- [ ] 3.3 重复运行固定子集，量化固定 seed 下的可复现性并校准静态 schedule 充分性 tolerance 的严格、中等、宽松三档候选值
- [ ] 3.4 仅使用 train 数据校准夹爪开闭模态与 phase-proxy 阈值，并冻结 tolerance、phase 规则和 test split
- [ ] 3.5 验证 `ATTENTION_BACKEND=cudnn`、T5 CPU offload 和双卡拓扑，分别记录 cold-start 与 warmup 后稳态延迟
- [ ] 3.6 验证连续 episode 重放和随机 anchor 两种模式的 session/priming/reset/KV-cache 对齐，并确认首个多帧 warmup 请求不进入稳态延迟

## 4. Phase 1 固定预算配对 Pilot 与 H1 判定

- [ ] 4.1 构造 50 个 episode、每个最多 10 个非重叠 anchor 的分层诊断 manifest，并从未参与定参的 episode 构造保持自然阶段占比的锁定确认 manifest；分别报告 task/phase 样本量与采样权重
- [ ] 4.2 在 GPU 0-1、2-3、4-5、6-7 上分别启动 static-5、6、8、16 双卡服务，完成同一 manifest 的可恢复配对扫描
- [ ] 4.3 在固定预算扫描后追加现有 video-only 动态基线，并核对每个样本的实际调用数和 decision trace
- [ ] 4.4 按每档 tolerance 计算 `E_best`、可接受静态 schedule 集合、最低调用数可接受 schedule、`E_5-E_best`、`E_8-E_best`、成对 compute benefit 和 schedule-质量 Pareto；报告完整非嵌套 step mask
- [ ] 4.5 按 episode 做配对 bootstrap，报告总体与 phase/task 子集的 mean、P90、P95 和置信区间
- [ ] 4.6 先在分层诊断集分析异质性，再在自然分布确认集输出 H1 go/no-go；若计算收益无稳定异质性，明确停止后续 gate 工作并记录未执行条件

## 5. Phase 2 特征、基线与 H2 判定

- [ ] 5.1 在 H1 go 时实现 V、A、M、A+M 特征抽取；覆盖 action/video 收敛、相对位移、速度、加速度/jerk、减速、方向变化、roughness、路径低效度和夹爪事件特征
- [ ] 5.2 生成单特征与连续 compute benefit、最低调用数可接受静态 schedule、phase proxy 的分析图表，检查“大/快等于难”的反例和特征数值稳定性
- [ ] 5.3 只在 train/validation 上拟合并锁定阈值规则、逻辑回归和浅层决策树，保存特征版本、归一化统计和 gate 配置；仅能区分 phase、不能预测 compute benefit 的特征不得进入 gate
- [ ] 5.4 实现 always-5/6/8/16、matched-compute random、video-only 和 oracle 基线，在冻结 test 上公平比较
- [ ] 5.5 在分层诊断集和自然分布确认集分别报告分类诊断指标与 matched-compute Pareto，并区分“A 有效但 M 无效”“M 只能识别阶段”和“M/A+M 能解释计算收益”三种结论
- [ ] 5.6 输出 H2 go/no-go 报告；若候选不能优于 random 与 video-only，则停止在线 scheduler 集成并记录原因

## 6. 条件式 Action-Aware Scheduler

> **本轮状态：条件未满足，不执行。** 只有新的闭环证据推翻 H2 no-go 后才可恢复本节。

- [ ] 6.1 仅在自然分布确认集上 H2 go 时增加 `DIT_SCHEDULE_MODE=fixed|video|action|combined` 配置解析，未设置时严格复用旧环境变量语义
- [ ] 6.2 实现 embodiment-aware 有效动作 mask、action-flow/provisional-action 特征计算和近零范数保护
- [ ] 6.3 实现版本化 gate 配置加载与 checkpoint、动作 schema、特征版本兼容校验，不兼容时 fail-compute
- [ ] 6.4 将 action/combined 决策接入现有单次 16-step `should_run_model` 路径，保持每步 scheduler update 与 skip 时 video/action flow 复用
- [ ] 6.5 实现每请求 gate 状态重置、session 级 causal/KV-cache 保留、episode reset 时完整清理、最小调用数、最大连续 skip、异常回退和可选轻量 decision trace
- [ ] 6.6 添加 synthetic trace 单元测试，覆盖前两步强制 compute、run/skip、NaN/Inf、零范数、未知 embodiment、异常回退和状态隔离
- [ ] 6.7 添加端到端兼容测试，确认 action-aware 模式仍返回有限 `(24, 8)` 动作，旧客户端无需修改

## 7. Phase 3 真实在线回放与 H3 判定

> **本轮状态：条件未满足，不执行 adaptive gate。** 已完成的是精确 prefix-stop 因果诊断，不是一个已部署 gate。

- [ ] 7.1 从 Phase 2 锁定最多两个候选 gate，在冻结 test manifest 上执行真实 run/skip 推理，不用 full-16 trace 模拟结果代替
- [ ] 7.2 重新计算动作质量、实际 DiT 调用、gate 开销、稳态延迟、P95 和 contact-proxy 子集指标
- [ ] 7.3 与默认 always-8、matched-compute random 和 video-only 做 episode-paired bootstrap 与 Pareto 比较
- [ ] 7.4 按“调用减少至少 20%、主指标均值退化不超过 2%、P95/关键阶段退化不超过 5%”输出 H3 go/no-go
- [ ] 7.5 仅在 H3 go 时扩展到至少 500 个 episode，复核置信区间稳定性；否则保持 action-aware 默认关闭并记录失败项

## 8. 报告、文档与最终校验

- [ ] 8.1 生成包含 manifest、splits、samples、traces、summary 和中文报告的版本化结果目录，并检查所有结论可追溯到逐样本记录
- [ ] 8.2 在报告中明确区分物理阶段代理、静态 schedule compute benefit、真实在线 skip 收益、action convergence 和真实接触，不把 contact-proxy 写成传感器真值
- [ ] 8.3 文档化 smoke、四服务 pilot、追加 video-only、分析和在线回放命令，包括 cuDNN SDPA 与 GPU pair 配置
- [ ] 8.4 运行评估模块单元测试、scheduler 回归测试和代表性双卡推理 smoke test，记录通过项与未覆盖风险
- [ ] 8.5 运行 OpenSpec 严格校验并核对 proposal、两份 specs、design、tasks 与最终 go/no-go 分支一致
- [ ] 8.6 若 H3 go，形成后续闭环仿真或机器人 replay 的外部有效性协议；若 no-go，归档可复现实验与停止结论
