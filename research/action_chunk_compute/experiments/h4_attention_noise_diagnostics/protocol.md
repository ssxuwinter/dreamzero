# H4 协议：attention entropy 与 DiT 输出噪声诊断

## 问题

在已确认速度/阶段信号不足以预测额外 DiT 计算收益后，检验两个同次推理可得的内部模型信号：

1. attention entropy：简单动作可能注意力更集中、熵更低；复杂动作可能注意力更分散、熵更高。
2. DiT 输出噪声/flow 强度：输出噪声或 flow 幅度可能反映模型确定性，并与动作简单度或继续计算收益相关。

## 预注册假设

- H4a：action-token self-attention entropy 与动作复杂度正相关；预期与 `gt_motion_h8`、fine stage、prefix action delta 有正相关。
- H4b：DiT action output/noise magnitude 与动作简单度或确定性相关；若“越高越确定、动作越简单”成立，则 action-flow norm 应与 `gt_motion_h8`、预测动作运动幅度、prefix delta 负相关。
- H4c：若任一内部信号能用于 compute gate，它必须在 episode-held-out 设置下预测 `benefit_k_to_16 = E_k - E_16`，并优于已有 flow/convergence/geometry 组合。

## 数据与范围

- 复用 H3 的 12 episode × 7 chunk causal prefix trace：`experiments/h3_nested_provisional_compute/results/prefix_trace_12`。
- 权重路径：`/home/admin/.cache/DreamZero-DROID`。
- 数据路径：`/home/admin/.cache/DreamZero-DROID-Data`。
- 先用已有 `action_flows` 对 H4b 做离线探索；attention entropy 需要新增 opt-in 插桩后另跑小样本。

## 指标

### DiT 输出噪声 / flow

对每个 chunk、每个 prefix checkpoint k ∈ {2,4,8,16} 计算：

- action-flow RMS / L2 norm；
- action-flow 与前一步的 delta RMS；
- action-flow 相对 delta；
- 可选：按 8 个动作维、joint/gripper 分组的 RMS。

### Attention entropy

新增诊断时只采集标量，不保存完整 attention map：

- action query tokens 对当前合法 context 的 normalized entropy；
- 按 DiT step 聚合 mean/std/min/max；
- 优先采集 inference KV-cache 分支和 action block 分支。

## 判定标准

- 动作复杂度指标：Spearman |rho| ≥ 0.3 且方向与假设一致，episode bootstrap 不由单 episode 驱动。
- compute gate 指标：episode-held-out `benefit>0.002` AUROC 明显超过已有 H3 best（约 0.61），且 R2/Spearman 不为系统性负。
- 若只与阶段/运动相关但不能预测 benefit，则只能作为复杂度解释信号，不能作为“多算” gate。

## 确认性与探索性标记

- 复用已有 H3 action_flows 的 H4b 检验标记为探索性，因为协议晚于原始 trace。
- 新增 attention entropy 插桩后重新跑的小样本可作为本协议下的确认性 pilot；若结果积极，再扩展到完整 12 episode。

## 2026-07-22 确认性阶段（结果前锁定）

### E1：attention entropy 跨阶段（21 chunk，3 episode × 7 chunk，5 阶段全覆盖）

- 数据：`image/` 三 episode（1925/16001/50458，均属 H3 集合），采集 `kv_action_register` 路径 normalized entropy。
- H5a 预测：若“复杂动作熵高”成立，fine(pre_close+closing+hold+release) − free_open 的 entropy 差 > 0，bootstrap 95% CI 不含 0。
- H5b 预测：若 entropy 可用于 compute gate，entropy 与 benefit_k*_to_16 的 Spearman CI 不含 0 且方向稳定。
- 若两者都不成立：判 H5 attention entropy 为当前证据 no-go（阶段刻画与 gate 均不支持）。

### E2：benefit 标签可学性上界（council 要求，用 H3 84 chunk）

- oracle_current_error（e_k，作弊特征）episode-bootstrap AUROC CI 若显著 > 0.5：标签可学，flow 的失败为特异性失败。
- 若 oracle 也横跨 0.5：判标签不可学，所有负结果降级为 underpowered null。
- 预测（基于机制推理）：oracle 会显著 > 0.5，因为当前误差大的 chunk 更可能有正 benefit（回归均值）。

### 结论判定规则（锁定）

- flow/noise 假设（“噪声越高→越确定→动作越简单”）：方向检验用 flow RMS 与 gt_motion/stage 的关系。已知 H4b 显示 flow RMS 与动作幅度**正**相关（自由移动动作幅度反而大），假设的方向性表述在 DROID 上不成立。
- 最终结论必须同时报告：方向性、gate 可用性、统计功效三个层面。

## 预期产物

- `results/flow_noise_existing_trace/`：H4b 离线分析表与报告。
- `results/attention_entropy_pilot/`：插桩小样本 trace、分析表与报告。
- 更新 `findings.md` 和 `research-state.yaml`。
