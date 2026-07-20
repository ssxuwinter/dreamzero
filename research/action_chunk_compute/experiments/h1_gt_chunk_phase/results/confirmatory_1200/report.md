# H1 GT 动作 chunk 间阶段实验结果

- episode 数：1200
- 全部 chunk 数：11030
- 可用阶段 chunk 数：10674
- 阶段计数：`{"hold": 4316, "free_open": 2303, "closing": 1592, "pre_close": 1385, "release": 1078, "ambiguous": 356}`

## 主结果

| 任务 | 特征 | AUROC | AUPRC | Balanced Acc. |
|---|---|---:|---:|---:|
| fine vs free | joint_only | 0.6620 | 0.8582 | 0.6190 |
| fine vs free | gripper_only | 0.9417 | 0.9831 | 0.9104 |
| fine vs free | joint_gripper | 0.9523 | 0.9863 | 0.9096 |
| fine vs free | joint_gripper_transition | 0.9547 | 0.9872 | 0.9092 |
| pre-close vs free | joint_only | 0.6873 | 0.5458 | 0.6350 |
| pre-close vs free | gripper_only | 0.6995 | 0.6648 | 0.6885 |
| pre-close vs free | joint_gripper | 0.7869 | 0.7381 | 0.7127 |
| pre-close vs free | joint_gripper_transition | 0.8001 | 0.7473 | 0.7212 |

## Transition 相对 gripper-only 的配对增量

- fine vs free：`{"auroc": {"point": 0.012999384337289643, "ci95": [0.010069189020137783, 0.01600251086882448]}, "auprc": {"point": 0.004140881888739623, "ci95": [0.003349610630997013, 0.005019426398448015]}}`
- pre-close vs free：`{"auroc": {"point": 0.10064003787243436, "ci95": [0.08248332476229431, 0.11934654319131462]}, "auprc": {"point": 0.08245300620378193, "ci95": [0.06750499695012453, 0.09619507822870475]}}`

## 五分类

- macro-F1：0.8620
- balanced accuracy：0.8665

## 解释限制

该实验的阶段由 GT 夹爪构造，因此 gripper-only 的 fine/free 结果主要是 sanity check。真正检验师兄想法的是 pre-close held-out 结果、transition 的增量，以及后续 WAM 配对实验中的 compute-benefit 预测。这里不能得出任何“应该多算”的结论。
