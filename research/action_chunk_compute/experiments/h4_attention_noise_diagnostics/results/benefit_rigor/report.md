# Benefit 标签严谨性检查（council 要求）

- 样本：12 episode / 84 chunk；tolerance=0.002

## 1. Benefit 分布（正收益天花板）

|   calls |     mean |     std |      q05 |   median |     q95 |   frac_pos |   frac_gt_0.002 |   frac_gt_0.005 |   n_pos_0.002 |   mean_relative_to_e16 |   q95_relative_to_e16 |
|--------:|---------:|--------:|---------:|---------:|--------:|-----------:|----------------:|----------------:|--------------:|-----------------------:|----------------------:|
|       2 | -0.00357 | 0.00712 | -0.01254 | -0.00247 | 0.00404 |    0.2619  |         0.13095 |         0.04762 |            11 |               -0.16439 |               0.1857  |
|       4 | -0.00228 | 0.00551 | -0.00871 | -0.00168 | 0.00596 |    0.27381 |         0.13095 |         0.05952 |            11 |               -0.10485 |               0.27425 |
|       8 | -0.00055 | 0.00237 | -0.00429 | -0.00058 | 0.00302 |    0.36905 |         0.11905 |         0.03571 |            10 |               -0.02552 |               0.1391  |

## 2. Episode-bootstrap AUROC（benefit>0.002）

|   calls | predictor                |   auroc |   ci_low |   ci_high |   valid_draws |
|--------:|:-------------------------|--------:|---------:|----------:|--------------:|
|       2 | h3_action_geometry       |   0.597 |    0.399 |     0.77  |          4000 |
|       2 | h3_chunk_transition      |   0.507 |    0.312 |     0.716 |          4000 |
|       2 | h3_combined              |   0.493 |    0.299 |     0.688 |          4000 |
|       2 | h3_flow_only             |   0.407 |    0.178 |     0.666 |          4000 |
|       2 | h3_gt_phase_oracle       |   0.421 |    0.234 |     0.593 |          4000 |
|       2 | h3_lagged_previous_chunk |   0.553 |    0.397 |     0.707 |          4000 |
|       2 | h3_prefix_convergence    |   0.517 |    0.314 |     0.699 |          4000 |
|       2 | h4b_dit_flow_scalar      |   0.377 |    0.24  |     0.527 |          4000 |
|       2 | raw_pred_motion          |   0.666 |    0.461 |     0.829 |          4000 |
|       2 | raw_flow_rms             |   0.697 |    0.481 |     0.852 |          4000 |
|       2 | raw_flow_delta_rms       |   0.569 |    0.379 |     0.718 |          4000 |
|       2 | oracle_current_error     |   0.788 |    0.603 |     0.946 |          4000 |
|       4 | h3_action_geometry       |   0.537 |    0.338 |     0.759 |          4000 |
|       4 | h3_chunk_transition      |   0.6   |    0.34  |     0.796 |          4000 |
|       4 | h3_combined              |   0.609 |    0.372 |     0.828 |          4000 |
|       4 | h3_flow_only             |   0.396 |    0.197 |     0.576 |          4000 |
|       4 | h3_gt_phase_oracle       |   0.486 |    0.389 |     0.633 |          4000 |
|       4 | h3_lagged_previous_chunk |   0.564 |    0.368 |     0.745 |          4000 |
|       4 | h3_prefix_convergence    |   0.514 |    0.317 |     0.713 |          4000 |
|       4 | h4b_dit_flow_scalar      |   0.411 |    0.28  |     0.534 |          4000 |
|       4 | raw_pred_motion          |   0.685 |    0.529 |     0.85  |          4000 |
|       4 | raw_flow_rms             |   0.649 |    0.429 |     0.878 |          4000 |
|       4 | raw_flow_delta_rms       |   0.645 |    0.526 |     0.785 |          4000 |
|       4 | oracle_current_error     |   0.854 |    0.709 |     0.948 |          4000 |
|       8 | h3_action_geometry       |   0.526 |    0.403 |     0.674 |          3992 |
|       8 | h3_chunk_transition      |   0.541 |    0.338 |     0.901 |          3992 |
|       8 | h3_combined              |   0.557 |    0.394 |     0.727 |          3992 |
|       8 | h3_flow_only             |   0.393 |    0.247 |     0.571 |          3992 |
|       8 | h3_gt_phase_oracle       |   0.48  |    0.319 |     0.642 |          3992 |
|       8 | h3_lagged_previous_chunk |   0.547 |    0.32  |     0.774 |          3992 |
|       8 | h3_prefix_convergence    |   0.389 |    0.171 |     0.604 |          3992 |
|       8 | h4b_dit_flow_scalar      |   0.403 |    0.256 |     0.629 |          3992 |
|       8 | raw_pred_motion          |   0.618 |    0.491 |     0.823 |          3992 |
|       8 | raw_flow_rms             |   0.569 |    0.4   |     0.768 |          3992 |
|       8 | raw_flow_delta_rms       |   0.685 |    0.57  |     0.802 |          3992 |
|       8 | oracle_current_error     |   0.762 |    0.648 |     0.938 |          3992 |

## 3. Episode-bootstrap Spearman（连续 benefit）

|   calls | predictor                |   spearman |   ci_low |   ci_high |
|--------:|:-------------------------|-----------:|---------:|----------:|
|       2 | h3_action_geometry       |      0.249 |   -0.008 |     0.467 |
|       2 | h3_chunk_transition      |     -0.002 |   -0.16  |     0.161 |
|       2 | h3_combined              |      0.004 |   -0.221 |     0.214 |
|       2 | h3_flow_only             |     -0.222 |   -0.347 |    -0.035 |
|       2 | h3_gt_phase_oracle       |     -0.158 |   -0.327 |     0.056 |
|       2 | h3_lagged_previous_chunk |     -0.151 |   -0.324 |    -0     |
|       2 | h3_prefix_convergence    |     -0.137 |   -0.32  |     0.058 |
|       2 | h4b_dit_flow_scalar      |     -0.042 |   -0.181 |     0.1   |
|       2 | raw_pred_motion          |      0.019 |   -0.158 |     0.226 |
|       2 | raw_flow_rms             |      0.022 |   -0.191 |     0.241 |
|       2 | raw_flow_delta_rms       |     -0.03  |   -0.205 |     0.144 |
|       2 | oracle_current_error     |      0.153 |   -0.073 |     0.357 |
|       4 | h3_action_geometry       |      0.091 |   -0.172 |     0.357 |
|       4 | h3_chunk_transition      |      0.161 |   -0.062 |     0.38  |
|       4 | h3_combined              |      0.19  |   -0.076 |     0.446 |
|       4 | h3_flow_only             |      0.062 |   -0.14  |     0.249 |
|       4 | h3_gt_phase_oracle       |     -0.117 |   -0.325 |     0.151 |
|       4 | h3_lagged_previous_chunk |      0.138 |   -0.055 |     0.281 |
|       4 | h3_prefix_convergence    |      0.133 |   -0.043 |     0.289 |
|       4 | h4b_dit_flow_scalar      |      0.134 |   -0.034 |     0.278 |
|       4 | raw_pred_motion          |     -0.065 |   -0.299 |     0.155 |
|       4 | raw_flow_rms             |     -0.048 |   -0.283 |     0.187 |
|       4 | raw_flow_delta_rms       |     -0.093 |   -0.282 |     0.102 |
|       4 | oracle_current_error     |      0.076 |   -0.141 |     0.264 |
|       8 | h3_action_geometry       |     -0.142 |   -0.269 |    -0.015 |
|       8 | h3_chunk_transition      |     -0.077 |   -0.261 |     0.111 |
|       8 | h3_combined              |     -0.106 |   -0.303 |     0.086 |
|       8 | h3_flow_only             |      0.007 |   -0.106 |     0.151 |
|       8 | h3_gt_phase_oracle       |     -0.127 |   -0.348 |     0.111 |
|       8 | h3_lagged_previous_chunk |      0.056 |   -0.174 |     0.249 |
|       8 | h3_prefix_convergence    |     -0.027 |   -0.186 |     0.106 |
|       8 | h4b_dit_flow_scalar      |     -0.011 |   -0.126 |     0.131 |
|       8 | raw_pred_motion          |     -0.096 |   -0.251 |     0.035 |
|       8 | raw_flow_rms             |     -0.082 |   -0.228 |     0.051 |
|       8 | raw_flow_delta_rms       |     -0.125 |   -0.293 |     0.063 |
|       8 | oracle_current_error     |      0.031 |   -0.083 |     0.153 |

## 判读要点

- 若所有 CI 均横跨 0.5（AUROC）或 0（Spearman），包括 oracle_current_error，则标签本身在该样本量下不可学，flow 的 0.484 不能解释为特异性失败。
- 若 oracle_current_error（当前误差，作弊特征）显著 >0.5 而 flow 仍在 0.5 附近，则可下 flow-specific no-go。
- benefit 相对 e16 的量级（mean_relative_to_e16）指示即便完美 gate 可省的误差上限。
