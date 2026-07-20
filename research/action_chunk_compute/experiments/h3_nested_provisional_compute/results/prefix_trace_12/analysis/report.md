# H3 同一去噪轨迹 prefix-stop 因果实验

- episode：12
- 连续 chunk：84
- prefix-16 与正常 final 最大差：0

## Quality-compute 曲线

| Calls | H8 joint MAE | 95% CI | Pred. motion H8 |
|---:|---:|---:|---:|
| 1 | 0.01818 | [0.01552, 0.02158] | 0.01518 |
| 2 | 0.01817 | [0.01554, 0.02146] | 0.01221 |
| 4 | 0.01946 | [0.01652, 0.02323] | 0.01230 |
| 8 | 0.02119 | [0.01821, 0.02489] | 0.01362 |
| 12 | 0.02169 | [0.01865, 0.02586] | 0.01347 |
| 16 | 0.02174 | [0.01860, 0.02603] | 0.01345 |

Persistence baseline H8 joint MAE：0.02475
GT motion H8：0.01595

## 阶段额外计算收益（E_k - E_16）

| 阶段 | 2→16 | 4→16 | 8→16 |
|---|---:|---:|---:|
| free_open | -0.00185 | -0.00111 | -0.00103 |
| pre_close | -0.00286 | -0.00162 | 0.00003 |
| closing | -0.00340 | -0.00292 | -0.00101 |
| hold | -0.00420 | -0.00265 | -0.00058 |
| release | -0.00573 | -0.00402 | -0.00042 |

## Full-16 非劣的最早 prefix oracle

| tolerance | Mean calls | <=2 | <=4 | <=8 | >8 |
|---:|---:|---:|---:|---:|---:|
| 0.000 | 2.93 | 77.4% | 85.7% | 89.3% | 10.7% |
| 0.002 | 1.74 | 86.9% | 91.7% | 96.4% | 3.6% |
| 0.005 | 1.26 | 95.2% | 96.4% | 98.8% | 1.2% |

## Held-out compute-benefit 预测

- k=2 / `flow_only`：Spearman -0.222，R2 -0.057，benefit>0.002 AUROC 0.407。
- k=2 / `prefix_convergence`：Spearman -0.137，R2 -0.040，benefit>0.002 AUROC 0.517。
- k=2 / `action_geometry`：Spearman 0.249，R2 0.042，benefit>0.002 AUROC 0.597。
- k=2 / `chunk_transition`：Spearman -0.002，R2 -0.133，benefit>0.002 AUROC 0.507。
- k=2 / `combined`：Spearman 0.004，R2 -0.436，benefit>0.002 AUROC 0.493。
- k=2 / `lagged_previous_chunk`：Spearman -0.151，R2 -0.695，benefit>0.002 AUROC 0.553。
- k=2 / `gt_phase_oracle`：Spearman -0.158，R2 -0.108，benefit>0.002 AUROC 0.421。
- k=4 / `flow_only`：Spearman 0.062，R2 -0.088，benefit>0.002 AUROC 0.396。
- k=4 / `prefix_convergence`：Spearman 0.133，R2 0.001，benefit>0.002 AUROC 0.514。
- k=4 / `action_geometry`：Spearman 0.091，R2 -0.078，benefit>0.002 AUROC 0.537。
- k=4 / `chunk_transition`：Spearman 0.161，R2 -0.046，benefit>0.002 AUROC 0.600。
- k=4 / `combined`：Spearman 0.190，R2 -0.046，benefit>0.002 AUROC 0.609。
- k=4 / `lagged_previous_chunk`：Spearman 0.138，R2 -0.172，benefit>0.002 AUROC 0.564。
- k=4 / `gt_phase_oracle`：Spearman -0.117，R2 -0.124，benefit>0.002 AUROC 0.486。
- k=8 / `flow_only`：Spearman 0.007，R2 -0.090，benefit>0.002 AUROC 0.393。
- k=8 / `prefix_convergence`：Spearman -0.027，R2 -0.051，benefit>0.002 AUROC 0.389。
- k=8 / `action_geometry`：Spearman -0.142，R2 -0.189，benefit>0.002 AUROC 0.526。
- k=8 / `chunk_transition`：Spearman -0.077，R2 -0.320，benefit>0.002 AUROC 0.541。
- k=8 / `combined`：Spearman -0.106，R2 -0.613，benefit>0.002 AUROC 0.557。
- k=8 / `lagged_previous_chunk`：Spearman 0.056，R2 -0.503，benefit>0.002 AUROC 0.547。
- k=8 / `gt_phase_oracle`：Spearman -0.127，R2 -0.086，benefit>0.002 AUROC 0.480。

## 运动幅度敏感性

GT motion 最高三分位中，2→16 mean benefit 为 -0.00083，正收益比例 46.4%。

## 判定

动作输出可以识别自由移动与精细操作阶段，但阶段、动作几何、chunk 间变化和 GT phase oracle 均未证明能够预测继续执行更多 DiT 的收益。当前离线 MAE 甚至随调用数增加而上升，因此不能据此实现“精细阶段多算”。

## 限制

DROID 只有成功示范，离线单轨迹 MAE 会惩罚不同但同样可行的多模态动作，也可能偏好保守动作。prefix-stop 结论足以否定当前 MAE 下的阶段门控依据，但不能证明 1-2 call 闭环成功率优于 full-16；上线前仍需仿真或真机闭环评估。
