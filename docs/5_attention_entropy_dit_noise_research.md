# Attention Entropy 与 DiT 输出噪声能否指示动作复杂度与计算收益（H5 研究报告）

日期：2026-07-22
模型：DreamZero-DROID（14B WAM，`/home/admin/.cache/DreamZero-DROID`）
预注册：protocol commit `5052501` → results commit `400b350`（协议先于结果提交）
工作区：`research/action_chunk_compute/experiments/h4_attention_noise_diagnostics/`

## 1. 研究问题

在前序研究（`docs/2_action_phase_compute_research.md`）确认"动作阶段/速度信号不能预测额外 DiT 计算收益"之后，检验两个新的模型内部信号假设：

- **H5a（attention entropy）**：简单动作时注意力更集中、熵更低；复杂动作时注意力更分散、熵更高。
- **H5b（DiT 输出噪声/flow）**：输出噪声越高，模型越确定，动作越简单；反之亦然。

以及衍生问题：这两个信号能否用于"当前 chunk 是否值得继续多算几步 DiT"的在线 gate。

## 2. 一句话结论

| 假设 | 裁定 | 关键证据 |
|---|---|---|
| 简单动作熵低、复杂动作熵高 | **不成立（确认性 no-go）** | 5 个阶段 entropy 全部 0.553±0.005，差异 <0.5%，CI 含 0 |
| 噪声越高→越确定→动作越简单 | **方向相反** | flow 幅度与动作幅度**正**相关（Spearman 0.91）；自由移动 flow 反而更大 |
| （衍生）flow 还在变→值得继续算 | **弱支持但不可部署** | AUROC 0.685 [0.57, 0.80]；但收益均值为负、正例仅 13% |

## 3. 实验设计

### 3.1 Attention entropy 采集（E1，确认性）

- 在 `groot/vla/model/dreamzero/modules/wan_video_dit_action_casual_chunk.py` 中新增默认关闭的探针（`TRACE_ATTENTION_ENTROPY=True`）：对 KV-cache 推理路径的 **action-register query** 计算对合法 context 的 normalized softmax entropy，只记标量（mean/std/min/max），按 8-head 分块避免 OOM，不保存 attention map。
- 数据：`image/` 下 3 个 episode（1925/16001/50458）的三视角真实视频，每个 episode 7 个 chunk，**5 个操作阶段全覆盖**（阶段标签来自 H1 的 1200-episode GT 夹爪迟滞标注，非模型自评）。
- 每个 chunk 产生 640 条记录（16 DiT call × 40 层），共 21 chunk。
- 双卡起真实 14B 推理服务采集（`socket_test_optimized_AR.py --port 8124`，full 16-call）。

### 3.2 DiT flow/noise 分析（H4b + E2 严谨性复核）

- 复用 H3 的 12 episode × 7 chunk causal prefix trace 中每步的 action flow 张量。
- 计算 flow RMS（幅度）与 flow delta RMS（收敛速度），按 all32/action8/joint7/gripper 维度分组。
- Council 评审后补做：benefit 分布、episode-bootstrap AUROC/Spearman 置信区间、oracle 上界（用"当前误差"作弊特征标定标签可学性）、不经拟合的单特征基线。

## 4. 结果

### 4.1 Attention entropy 跨阶段完全平坦

| 阶段 | chunk 数 | entropy 均值 |
|---|---|---|
| free_open | 4 | 0.5533 |
| pre_close | 3 | 0.5534 |
| closing | 3 | 0.5546 |
| hold | 8 | 0.5551 |
| release | 3 | 0.5526 |

- fine(后四阶段) − free 差 = **+0.00095，95% CI [−0.00148, +0.00373]**，含 0。
- 三个 episode 内部方向一致为正（+0.0006~+0.0017），但效应量只有 chunk 内记录间离散度（0.136）的 **1/50**——即使方向真实也无实用区分度。
- 与 benefit_k2/k4/k8 的 Spearman = +0.08/+0.04/−0.13，p 全部 >0.58。
- 与 gt_motion_h8 的 Spearman = 0.18（p=0.44）。
- 混杂检查：entropy 与 anchor/KV 长度无关（rho=0.02，p=0.92）。

**判定：H5a 按预注册规则 no-go。** 机器人明明经历了完全不同的操作阶段（GT 可证），模型 action attention 的集中程度却纹丝不动——阶段信息不体现在注意力熵上。

### 4.2 DiT flow/noise：方向相反 + 一个真实弱信号

方向性：

- `k2_dit_flow_joint7_rms` vs `k2_pred_motion_h8` Spearman **0.906**：flow 幅度基本就是动作幅度的镜像。自由移动这类"简单"动作幅度大、flow 反而高；精细操作动作小、flow 反而低。"噪声高=更确定=更简单"在 DROID 上方向相反。
- `k8_dit_flow_delta_joint7_rms` vs `k8_prefix_joint_delta_mean` Spearman **0.985**：flow delta 与 prefix 收敛量是同一信号的两种测量（AUROC 也完全相同）。

gate 可用性（benefit>0.002 的 episode-bootstrap AUROC，k=8）：

| 预测信号 | AUROC | 95% CI | 判定 |
|---|---|---|---|
| oracle：当前误差（作弊上界） | 0.762 | [0.648, 0.938] | 标签可学 |
| **raw flow delta RMS（收敛速度）** | **0.685** | **[0.570, 0.802]** | 真实弱信号 |
| raw 预测动作幅度 | 0.618 | [0.491, 0.823] | 边缘 |
| ridge 拟合 flow 组合 | 0.403 | [0.256, 0.629] | 小样本过拟合失败 |
| GT 阶段 oracle | 0.480 | [0.319, 0.642] | 阶段无信息 |

收益天花板：

- benefit 均值在 k=2/4/8 全为**负**（−0.0036/−0.0023/−0.0006）：多算平均反而更差。
- 正收益 chunk 仅 13%（11/84）；q95 收益相对 e16 平均误差只有 14-27%。
- 连续 benefit 的所有 Spearman CI（含 oracle）横跨 0：只有二值尾部可分。
- flow delta 信号只在 7 个真实关节维上存在（32 维全量稀释后 AUROC 降至 0.515）。

**判定：方向性假设不成立；收敛速度信号真实但达不到部署门槛（decision_gates 要求）。**

### 4.3 统计功效自检（回应 council 评审）

- 标签可学性由 oracle 标定：AUROC 0.76-0.85 且 CI 不含 0.5 → **负结果是特异性失败，不是样本不足的假阴性**。
- 早前"最佳 AUROC 0.484"的表述已修正：那是 ridge 拟合组合在小样本上过拟合翻符号的结果；不经拟合的原始收敛速度分数有 0.685 的真实信号。
- 预测在结果前锁定（git 预注册），二值/连续、多容差、拟合/不拟合全部报告。

## 5. 机制解释

1. **entropy 恒定**：该 WAM 的 action-register 注意力模式由架构与训练分布决定（看哪些图像 token、看多少 context），与当前任务阶段解耦。"难的动作需要看得更散"这一直觉在这个模型上不成立。
2. **flow ≈ 动作幅度**：flow matching 的输出本质是去噪方向向量，其范数由动作在归一化空间的位移决定，与"确定性"无直接关系。
3. **收敛 ≠ 收敛到正确答案**：模型可以自信地收敛到一个错误动作（低 flow delta、高误差），所以内部确定性类信号预测不了误差收益。唯一有效的"当前误差"信号在线上不可得（需要 GT）。

## 6. 局限

- entropy 确认性实验为 3 episode / 21 chunk；阶段平坦性效应量清晰（噪声的 1/50），但发表级说服力应扩到 12 episode。
- benefit 全部基于离线单轨迹 MAE；DROID 成功示范存在多模态问题，离线 MAE 可能偏好保守动作。闭环成功率验证仍缺失（与前序研究相同的边界）。
- 单一 checkpoint、单一 embodiment（DROID）。

## 7. 建议

1. **停止在离线指标上继续挖内部信号**：oracle 已标定信号上限，entropy/flow/几何/阶段全部试过。
2. 最有价值的下一个实验是**闭环比较固定 2/4/8/16-call 的任务成功率**——离线 MAE 已显示 1-2 call 不劣于 16-call。
3. 若坚持自适应计算，唯一有依据的形态是用 flow-delta 做**保守 early-exit**（明显已收敛才提前停），并直接在闭环验证，不再做离线中间站。

## 8. 产物索引

| 内容 | 路径 |
|---|---|
| 预注册协议 | `experiments/h4_attention_noise_diagnostics/protocol.md` |
| entropy 确认性数据+分析 | `experiments/h4_attention_noise_diagnostics/results/attention_entropy_stages_21/` |
| benefit 严谨性检查 | `experiments/h4_attention_noise_diagnostics/results/benefit_rigor/` |
| flow/noise 离线分析 | `experiments/h4_attention_noise_diagnostics/results/flow_noise_existing_trace/` |
| 采集/分析脚本 | `research/action_chunk_compute/src/run_attention_entropy_image_pilot.py`、`analyze_attention_entropy_stages.py`、`analyze_benefit_rigor.py`、`analyze_dit_flow_noise.py` |
| 模型插桩 | `groot/vla/model/dreamzero/modules/wan_video_dit_action_casual_chunk.py`（`TRACE_ATTENTION_ENTROPY`） |
| 可视化报告 | `research/action_chunk_compute/to_human/h5_final_report_20260722.html` |
