# H2 WAM 静态 5/8/16-call 连续 chunk 配对结果

- episode：12
- 连续 chunk：84
- 阶段计数：`{"hold": 26, "free_open": 22, "pre_close": 12, "closing": 12, "release": 12}`

## Schedule 总体

| Calls | Joint MAE H8 (q99 norm) | Joint MAE H24 | Gripper MAE H8 | Warm latency |
|---:|---:|---:|---:|---:|
| 5 | 0.02108 | 0.03437 | 0.10422 | 2.70s |
| 8 | 0.02121 | 0.03389 | 0.09177 | 3.92s |
| 16 | 0.02174 | 0.03506 | 0.09226 | 7.20s |

## 额外计算收益

- `benefit_5_to_8`：均值 -0.00013，95% CI [-0.00045, 0.00018]；正收益比例 56.0%，>0.002 比例 8.3%。
- `benefit_5_to_16`：均值 -0.00066，95% CI [-0.00167, 0.00049]；正收益比例 47.6%，>0.002 比例 20.2%。
- `benefit_8_to_16`：均值 -0.00053，95% CI [-0.00140, 0.00048]；正收益比例 38.1%，>0.002 比例 16.7%。

## 阶段与收益

| 阶段 | 5→16 mean benefit | 正收益比例 |
|---|---:|---:|
| free_open | 0.00020 | 54.5% |
| pre_close | -0.00176 | 50.0% |
| closing | -0.00011 | 41.7% |
| hold | 0.00036 | 50.0% |
| release | -0.00316 | 33.3% |

## 阶段识别（episode-held-out）

- `fine_vs_free` / `current_joint`：AUROC 0.689，AUPRC 0.833。
- `fine_vs_free` / `current_gripper`：AUROC 0.911，AUPRC 0.972。
- `fine_vs_free` / `current_joint_gripper_transition`：AUROC 0.891，AUPRC 0.961。
- `fine_vs_free` / `lagged_previous_chunk`：AUROC 0.764，AUPRC 0.885。
- `pre_close_vs_free` / `current_joint`：AUROC 0.833，AUPRC 0.720。
- `pre_close_vs_free` / `current_gripper`：AUROC 0.447，AUPRC 0.371。
- `pre_close_vs_free` / `current_joint_gripper_transition`：AUROC 0.678，AUPRC 0.504。
- `pre_close_vs_free` / `lagged_previous_chunk`：AUROC 0.650，AUPRC 0.539。

## Compute-benefit held-out 预测

- `current_joint`：Spearman -0.031，R2 -0.157，benefit>0.002 AUROC 0.452。
- `current_gripper`：Spearman -0.026，R2 -0.043，benefit>0.002 AUROC 0.429。
- `current_joint_gripper_transition`：Spearman 0.056，R2 -0.176，benefit>0.002 AUROC 0.461。
- `lagged_previous_chunk`：Spearman -0.062，R2 -0.306，benefit>0.002 AUROC 0.388。
- `gt_phase_oracle`：Spearman -0.181，R2 -0.065，benefit>0.002 AUROC 0.327。

## 最低可接受静态 schedule（oracle 诊断）

| tolerance | mean calls | s5 | s8 | s16 |
|---:|---:|---:|---:|---:|
| 0.000 | 10.14 | 23.8% | 40.5% | 35.7% |
| 0.002 | 6.74 | 73.8% | 14.3% | 11.9% |
| 0.005 | 5.26 | 97.6% | 0.0% | 2.4% |

## 解释限制

5/8/16-call mask 不嵌套，本实验的 benefit 是静态 schedule-policy 差异，不是从同一中间状态继续计算的因果收益。当前最终输出可用于下一 chunk 的 lagged 调度；只有 H3 的 provisional-action/action-flow 轨迹才能决定当前请求是否早停。
