# H5 attention entropy 跨阶段分析（21 chunk / 3 episode / 5 stage）

- attn 路径：kv_action_register（实际推理路径）；每 chunk 记录数 ≈ 640

## 按阶段的 entropy

| stage     |   chunks |   entropy_mean |   entropy_sd |   gt_motion |
|:----------|---------:|---------------:|-------------:|------------:|
| free_open |        4 |        0.55332 |      0.002   |     0.01985 |
| pre_close |        3 |        0.55336 |      0.0023  |     0.01581 |
| closing   |        3 |        0.55459 |      0.00607 |     0.00796 |
| hold      |        8 |        0.55512 |      0.00475 |     0.01705 |
| release   |        3 |        0.55257 |      0.0039  |     0.00775 |

## fine(pre_close+closing+hold+release) − free_open 差值

diff = 0.00095, 95% CI [-0.00148, 0.00373]

## entropy 与复杂度/收益的 Spearman

| target            |   n |   spearman |      p |
|:------------------|----:|-----------:|-------:|
| gt_motion_h8      |  21 |     0.1792 | 0.437  |
| k8_pred_motion_h8 |  21 |    -0.1714 | 0.4575 |
| benefit_k2_to_16  |  21 |     0.0753 | 0.7456 |
| benefit_k4_to_16  |  21 |     0.0416 | 0.858  |
| benefit_k8_to_16  |  21 |    -0.1273 | 0.5825 |
| e16_joint_h8      |  21 |    -0.3169 | 0.1616 |

## 判读

- 假设 H5a 预测 fine 阶段 entropy 更高。若 CI 含 0 或方向相反，则该样本不支持。
- 若 entropy 与 benefit 的相关 CI 含 0，则 entropy 也不能作为 compute gate 信号。
