# H4b DiT 输出 noise/flow 既有 trace 分析

- episode：12
- chunk：84
- 数据：H3 full-16 causal prefix trace 中已有 `action_flows`，因此本轮为探索性离线分析。

## 与动作复杂度的 Spearman 相关

| Calls | Feature | Target | Spearman | p |
|---:|---|---|---:|---:|
| 8 | `k8_dit_flow_delta_joint7_rms` | `k8_prefix_joint_delta_mean` | 0.985 | 6.7e-65 |
| 4 | `k4_dit_flow_delta_joint7_rms` | `k4_prefix_joint_delta_mean` | 0.984 | 1.28e-63 |
| 2 | `k2_dit_flow_delta_joint7_rms` | `k2_prefix_joint_delta_mean` | 0.981 | 9.43e-60 |
| 2 | `k2_dit_flow_joint7_rms` | `k2_pred_motion_h8` | 0.906 | 2.65e-32 |
| 4 | `k4_dit_flow_joint7_rms` | `k4_pred_motion_h8` | 0.899 | 4.33e-31 |
| 1 | `k1_dit_flow_joint7_rms` | `k2_pred_motion_h8` | 0.886 | 4.46e-29 |
| 8 | `k8_dit_flow_joint7_rms` | `k8_pred_motion_h8` | 0.871 | 4.44e-27 |
| 2 | `k2_dit_flow_joint7_rms` | `k4_pred_motion_h8` | 0.871 | 5.26e-27 |
| 4 | `k4_dit_flow_joint7_rms` | `k2_pred_motion_h8` | 0.870 | 5.86e-27 |
| 8 | `k8_dit_flow_joint7_rms` | `k4_pred_motion_h8` | 0.869 | 8.13e-27 |
| 16 | `k16_dit_flow_joint7_rms` | `k4_pred_motion_h8` | 0.859 | 1.53e-25 |
| 16 | `k16_dit_flow_joint7_rms` | `k8_pred_motion_h8` | 0.853 | 7.12e-25 |
| 1 | `k1_dit_flow_joint7_rms` | `k4_pred_motion_h8` | 0.837 | 3.63e-23 |
| 16 | `k16_dit_flow_all32_rms` | `k8_pred_motion_h8` | 0.817 | 2.46e-21 |
| 16 | `k16_dit_flow_all32_rms` | `k4_pred_motion_h8` | 0.811 | 8.18e-21 |
| 8 | `k8_dit_flow_all32_rms` | `k8_pred_motion_h8` | 0.790 | 4.05e-19 |
| 4 | `k4_dit_flow_joint7_rms` | `k8_pred_motion_h8` | 0.790 | 4.52e-19 |
| 16 | `k16_dit_flow_joint7_rms` | `k2_pred_motion_h8` | 0.785 | 9.64e-19 |
| 8 | `k8_dit_flow_all32_mean_abs` | `k8_pred_motion_h8` | 0.783 | 1.44e-18 |
| 16 | `k16_dit_flow_all32_mean_abs` | `k4_pred_motion_h8` | 0.782 | 1.54e-18 |

## 按阶段的 action-flow RMS

| Calls | Stage | n | action8 RMS | joint7 RMS | delta action8 RMS | GT motion |
|---:|---|---:|---:|---:|---:|---:|
| 8 | closing | 12 | 1.0323 | 0.9979 | 0.0201 | 0.0064 |
| 8 | free_open | 22 | 1.0882 | 1.0437 | 0.0157 | 0.0197 |
| 8 | hold | 26 | 1.0627 | 1.0312 | 0.0252 | 0.0173 |
| 8 | pre_close | 12 | 1.0606 | 1.0211 | 0.0202 | 0.0163 |
| 8 | release | 12 | 1.0498 | 1.0137 | 0.0279 | 0.0152 |

## Held-out compute-benefit 预测

- k=2 / `dit_flow_scalar`：Spearman -0.042，R2 -0.050，benefit>0.002 AUROC 0.377，AUPRC 0.109。
- k=2 / `dit_flow_expanded`：Spearman -0.052，R2 -0.150，benefit>0.002 AUROC 0.465，AUPRC 0.146。
- k=4 / `dit_flow_scalar`：Spearman 0.134，R2 0.026，benefit>0.002 AUROC 0.411，AUPRC 0.113。
- k=4 / `dit_flow_expanded`：Spearman 0.157，R2 0.033，benefit>0.002 AUROC 0.484，AUPRC 0.128。
- k=8 / `dit_flow_scalar`：Spearman -0.011，R2 -0.151，benefit>0.002 AUROC 0.403，AUPRC 0.118。
- k=8 / `dit_flow_expanded`：Spearman 0.104，R2 -0.152，benefit>0.002 AUROC 0.453，AUPRC 0.118。

## 初步判定

最强复杂度相关为 `k8_dit_flow_delta_joint7_rms` vs `k8_prefix_joint_delta_mean`，Spearman 0.985。
最佳 compute-benefit AUROC 为 k=4 / `dit_flow_expanded` 的 0.484。
若方向与假设不一致或不能超过 H3 既有 flow/convergence 结果，则 DiT 输出 flow 只能作为弱诊断，不足以支持在线多算 gate。
