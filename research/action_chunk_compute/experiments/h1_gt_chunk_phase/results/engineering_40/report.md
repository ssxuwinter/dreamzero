# H1 GT 动作 chunk 间阶段实验结果

- episode 数：40
- 全部 chunk 数：377
- 可用阶段 chunk 数：361
- 阶段计数：`{"hold": 139, "free_open": 87, "closing": 54, "pre_close": 43, "release": 38, "ambiguous": 16}`

## 主结果

| 任务 | 特征 | AUROC | AUPRC | Balanced Acc. |
|---|---|---:|---:|---:|
| fine vs free | joint_only | 0.6199 | 0.8225 | 0.5662 |
| fine vs free | gripper_only | 0.9451 | 0.9832 | 0.9356 |
| fine vs free | joint_gripper | 0.9502 | 0.9864 | 0.9374 |
| fine vs free | joint_gripper_transition | 0.9532 | 0.9869 | 0.9123 |
| pre-close vs free | joint_only | 0.5809 | 0.4139 | 0.5612 |
| pre-close vs free | gripper_only | 0.6600 | 0.6265 | 0.6804 |
| pre-close vs free | joint_gripper | 0.6431 | 0.6292 | 0.5891 |
| pre-close vs free | joint_gripper_transition | 0.6889 | 0.6388 | 0.6065 |

## Transition 相对 gripper-only 的配对增量

- fine vs free：`{"auroc": {"point": 0.008096316805100923, "ci95": [-0.008776001999325309, 0.03279783629782407]}, "auprc": {"point": 0.0036886279160717184, "ci95": [-0.0007199416200750262, 0.009189438838335744]}}`
- pre-close vs free：`{"auroc": {"point": 0.02886928628708907, "ci95": [-0.07844113766668126, 0.11750425075009846]}, "auprc": {"point": 0.012304828803081591, "ci95": [-0.08778928843779546, 0.08238690629354807]}}`

## 五分类

- macro-F1：0.8027
- balanced accuracy：0.8135

## 解释限制

该实验的阶段由 GT 夹爪构造，因此 gripper-only 的 fine/free 结果主要是 sanity check。真正检验师兄想法的是 pre-close held-out 结果、transition 的增量，以及后续 WAM 配对实验中的 compute-benefit 预测。这里不能得出任何“应该多算”的结论。
