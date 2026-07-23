# 研究发现

## H4 新增内部指标初筛（2026-07-22）

用户提出两个新候选指标：attention entropy 与 DiT 输出噪声/flow 强度。先复用 H3 full-16 prefix trace 中已有 `action_flows` 做探索性分析。

### DiT 输出 noise/flow

- 12 episode、84 chunk 的既有 trace 上，action-flow RMS 与模型自己预测的动作运动幅度强相关：例如 `k2_dit_flow_joint7_rms` vs `k2_pred_motion_h8` Spearman 0.906。
- flow delta 与 prefix action 收敛量几乎等价：`k8_dit_flow_delta_joint7_rms` vs `k8_prefix_joint_delta_mean` Spearman 0.985。这说明它确实是“内部去噪变化量”的良好观测。
- 但它不能预测继续计算收益：episode-held-out 下最佳 `benefit>0.002` AUROC 只有 0.484，低于 H3 已有 flow/convergence 组合约 0.609，也低于可用 gate 的证据门槛。
- 阶段均值也不支持“越高越确定、动作越简单”的单调解释：k=8 action-flow RMS 在 free_open 反而略高于 closing。

结论：DiT action-flow/noise 强度可以作为动作幅度和去噪收敛的解释性诊断，但当前证据不支持用它决定“是否继续多算”。

### Benefit 标签严谨性复核（2026-07-22，council 意见后补做）

对 84 chunk 做了 benefit 分布、episode-bootstrap AUROC/Spearman CI 和 oracle 上界检查（`results/benefit_rigor/`），修正三个先前不精确的说法：

1. **标签可学，不是纯噪声**：作弊特征 oracle_current_error（当前 k-call 误差本身）AUROC = 0.79/0.85/0.76（k=2/4/8），95% CI 全部不含 0.5。benefit>0.002 标签有真实结构，flow 拟合组合的失败是特异性失败，不能全部归咎统计功效。
2. **拟合组合失败，但单特征原始分数有弱信号**：ridge 拟合的 flow 组合 AUROC 0.38-0.41；而**不经拟合**的 `raw_flow_delta_rms`（k=8）AUROC 0.685，CI [0.57, 0.80] 不含 0.5——方向为“去噪未收敛的 chunk 更可能从多算获益”，机制自洽。小样本下线性拟合多特征反而过拟合翻符号。
3. **收益天花板很低**：benefit 均值所有 k 均为负（多算平均更差），正收益 chunk 仅 13%（11/84），q95 收益相对 e16 平均误差 14-27%；连续 benefit 的所有 Spearman CI（含 oracle）横跨 0，只有二值尾部可分。

修正后的 flow/noise 结论：**方向性**——“噪声/flow 越高→越确定→动作越简单”在 DROID 上不成立（flow RMS 与动作幅度正相关，自由移动幅度更大）；**gate 可用性**——收敛速度维度（delta RMS，k=8）有真实弱信号 AUROC≈0.69，但收益天花板低、正例稀少，达不到部署门槛；**统计功效**——标签本身可学（oracle 显著），负结果是特异性的而非 underpowered。

### Attention entropy

已新增默认关闭的 `TRACE_ATTENTION_ENTROPY=True` 诊断插桩，只记录 action-register 查询对合法 KV context 的 normalized entropy 标量（OOM 修复后语义：仅 action query，按 head 分块）。

**确认性实验（2026-07-22，21 chunk / 3 episode / 5 阶段全覆盖，协议先行锁定于 commit 5052501）**：

- 按阶段 entropy 均值：free_open 0.5533、pre_close 0.5534、closing 0.5546、hold 0.5551、release 0.5526——几乎完全平坦。
- H5a（fine 阶段熵更高）：fine−free 差 = **+0.00095，95% CI [−0.00148, +0.00373]，含 0，不显著**。方向在 3 个 episode 内部均为正（+0.0006~+0.0017），但效应量只有 chunk 内 record 间离散度（0.136）的约 1/50，即使方向真实也无实用区分度。
- H5b（entropy 预测 benefit）：与 benefit_k2/k4/k8 的 Spearman 分别为 +0.08/+0.04/−0.13，p 均 >0.58，全不显著。
- 混杂检查：entropy 与 anchor/KV 长度的相关 rho=0.02（p=0.92），无位置混杂。
- 与动作复杂度的相关也不成立：vs gt_motion_h8 rho=0.18（p=0.44）。

**H5 attention entropy 结论：no-go（确认性）**。在 chunk 级别，action attention entropy 既不区分简单/复杂动作阶段，也不预测多算收益；模型对 action token 的注意力分布集中程度在整条轨迹上高度恒定（0.553±0.005）。原始直觉“简单动作熵低、复杂动作熵高”在 DreamZero-DROID 上不成立。

## 最终判定

状态：**阶段识别 go；按阶段多算 no-go。**

师兄的想法包含两个命题，实验只支持第一个：

1. **支持**：可以从动作 chunk 的夹爪事件、关节减速/几何和相邻 chunk 变化中识别自由移动与精细操作阶段。
2. **不支持**：精细阶段没有表现出更高的额外 DiT 计算收益；动作几何、chunk 间变化、action-flow 收敛和 GT phase oracle 都不能可靠预测“继续多算是否更好”。

因此不能实现“自由移动少算、精细操作多算”的阶段感知 gate。阶段分类器仍可用于轨迹分析或自适应执行 chunk 长度，但那是另一个应用。

## 三层证据

### 1. GT 动作阶段，1,200 episode

- 11,030 个连续 chunk，10,674 个具有清晰阶段代理。
- `pre-close vs free`：joint-only AUROC 0.687，gripper-only 0.699，joint+gripper 0.787，加入 transition 后 0.800。
- transition 相对 joint+gripper 的独立 AUROC 增量为 0.0132，episode bootstrap 95% CI `[0.0038, 0.0225]`。
- closing 的关节平均步长、加速度和 jerk 中位数均低于 free；“越大、越快、变化越剧烈越难”不成立。

### 2. WAM 静态 5/8/16-call，12 episode x 7 chunk

- H8 q99-normalized joint MAE：5-call 0.02108，8-call 0.02121，16-call 0.02174。
- `E5-E16` 均值 -0.00066，95% CI `[-0.00167, 0.00049]`；没有总体正收益。
- WAM predicted-gripper 能识别 fine/free，episode-held-out AUROC 0.911。
- 同一批动作特征预测 `benefit>0.002` 的 AUROC 仅 0.388-0.461；GT phase oracle 为 0.327。

### 3. 同轨迹 prefix-stop，12 episode x 7 chunk

- prefix-16 与正常 full-16 逐元素一致，最大差 0，排除了不嵌套静态 mask 的混杂。
- H8 joint MAE 随调用数总体上升：1-call 0.01818，2-call 0.01817，4-call 0.01946，8-call 0.02119，16-call 0.02174。
- 各阶段 `E2-E16` 均为负：free -0.00185、pre-close -0.00286、closing -0.00340、hold -0.00420、release -0.00573。
- flow、provisional convergence、动作几何、transition、lagged previous chunk 和 GT phase 的 held-out R2 几乎全部为负；`benefit>0.002` 最佳 AUROC 约 0.609，不满足 gate 证据要求。

## 对“100 步动作”的解释

若一条轨迹有 100 步，可以把它按连续窗口分成自由移动、pre-close、closing、hold、release。前段常表现为较大的路径和速度，后段常表现为减速、平滑、夹爪切换与方向调整。相邻窗口统计量确实提供边界信息。

但是阶段边界只能说明动作语义发生变化，不能说明扩散模型继续去噪会更接近该条示范。在当前 DreamZero/DROID 离线指标下，两者没有建立关系。

## 关键边界

- 最终 chunk 输出只能调度下一 chunk；当前请求早停必须使用当前 action flow 或 provisional action。
- action-only 无法观测爪尖到物体距离、真实接触力或滑移。
- DROID 单条成功示范具有多模态问题，离线 MAE 可能偏好保守动作；不能据此宣称 1-2 call 的闭环成功率更高。
- 下一步若追求加速，应先在闭环任务成功率上比较固定 2/4/8-call，而不是实现阶段感知多算 gate。

