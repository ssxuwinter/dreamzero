# 研究日志

## 2026-07-19：研究初始化

- 将师兄的直觉拆成两个独立命题：动作输出能否识别精细操作阶段；该阶段或信号能否预测继续执行 DiT 的收益。
- 锁定三个层次的实验：GT 动作大样本阶段分析、WAM 静态 5/8/16 配对推理、单条 nested 去噪轨迹的 provisional-action 早停实验。
- 明确最终 chunk 输出只能用于下一 chunk 的滞后决策；若要决定当前 chunk 是否继续多算，必须使用当前推理中已经产生的 action-flow 或 provisional action。
- 已有单 episode pilot 支持“输出动作含有阶段信号”，但不支持“交互阶段更值得多算”。本工作区从多 episode 证据重新检验。
- 当前仓库含用户未提交改动，因此所有模型结果必须记录 git HEAD、dirty 状态、环境变量和 manifest；在锁定协议后才运行确认性实验。

## 决策记录

| 日期 | 决策 | 原因 |
|---|---|---|
| 2026-07-19 | 夹爪 GT 只构造阶段代理，不作为在线 gate 输入 | 避免未来信息泄漏与循环定义 |
| 2026-07-19 | joint-only、predicted-gripper-only、joint+predicted-gripper+transition 分组比较 | 分辨阶段信息究竟来自夹爪事件还是关节几何 |
| 2026-07-19 | 将静态 5/8/16 结果标记为探索性 compute proxy | 三个 step mask 不嵌套，不能解释为纯粹“多算几步”的因果收益 |
| 2026-07-19 | 最终判断以 nested 当前轨迹为准 | 只有从同一个早期状态继续运行才能测量真正的当前请求额外计算收益 |

## 2026-07-19：H1 大样本 GT 阶段实验

- 按冻结随机种子抽取 1,200 个成功 episode，得到 11,030 个连续 24-step chunk。
- 用 episode 内夹爪迟滞阈值构造 free/open、pre-close、closing、hold、release；356 个不清晰 chunk 标记 ambiguous。
- episode-grouped 5-fold 结果表明动作几何和 chunk transition 能提前区分 pre-close；full 特征 AUROC 0.800。
- closing 相比 free 更慢、更平滑，支持“大小/变化率不是正单调难度指标”。

## 2026-07-19：H2 WAM 静态 schedule 实验

- 冻结 12 条阶段完整的连续轨迹，每条 7 个 chunk；三组双卡服务并行运行 static-5/8/16。
- 三个 schedule 的重复子集逐元素差为 0；cuDNN SDPA、T5 CPU offload 和双卡推理均正常。
- 5/8/16-call H8 joint MAE 分别为 0.02108/0.02121/0.02174；16-call 没有总体收益。
- predicted action 能识别阶段，但所有动作特征和 GT phase 均不能 held-out 预测静态 compute benefit。

## 2026-07-19：H3 同轨迹 prefix-stop 实验

- 新增默认关闭的研究 trace；在 full-16 中保存 8 个有效动作维的 flow，并离线重放 `k=1..16` prefix-stop。
- prefix-16 与正常 final 及独立 static-16 逐元素一致，证明回放对齐。
- 84 个 chunk 上，离线 H8 joint MAE 从 1/2-call 的约 0.0182 上升到 16-call 的 0.02174。
- pre-close 的 8→16 benefit 约 0.00003，其余阶段为负；没有“精细阶段更需要多算”的证据。
- compute-benefit predictor 全部未过门槛，研究按预注册停止规则判为 no-go，不继续实现在线 action-aware scheduler。

## 2026-07-22：H4 attention entropy 与 DiT flow/noise 新指标

- 根据用户和师兄的新想法，新增 H4：检验 attention entropy 与 DiT 输出 noise/flow 强度是否能刻画动作简单/复杂，或预测继续计算收益。
- 已锁定协议：`experiments/h4_attention_noise_diagnostics/protocol.md`。
- 复用 H3 full-16 prefix trace 中已有 `action_flows` 做探索性 H4b 分析，新增脚本 `src/analyze_dit_flow_noise.py`。
- H4b 结果：action-flow RMS 强烈跟随模型预测动作运动幅度，flow delta 强烈跟随 prefix 收敛量；但 episode-held-out compute-benefit 预测最佳 AUROC 只有 0.484，不支持作为多算 gate。
- 新增 opt-in attention entropy 插桩：`TRACE_ATTENTION_ENTROPY=True` 时记录 action/query attention 的 normalized entropy 标量，并经 policy/server/trace runner 传出。
- 用户提供 `image/` 下 3 个 episode 的三视角视频作为 attention entropy pilot 数据；新增 `src/run_attention_entropy_image_pilot.py`。
- 运行服务前发现 `dream` 环境同时安装 GUI OpenCV 和 headless OpenCV，GUI 版缺 `libgthread-2.0.so.0`；已移除 `opencv-python` 并重装 `opencv-python-headless==4.11.0.86`，cv2 import 恢复。
- 新启动服务还缺 `google/umt5-xxl` tokenizer 本地文件；权重目录有 T5/text encoder 权重，但 tokenizer 不在 DreamZero-DROID checkpoint 内。已通过 `HF_ENDPOINT=https://hf-mirror.com` 下载到 `/home/admin/.cache/google-umt5-xxl`，并用 `--tokenizer-path` 显式指定。
- `image/` pilot 已跑通 3 个 episode，每个 chunk 680 条 entropy record；三个样本均为 free_open，mean entropy 约 0.620，说明链路可用但样本没有阶段多样性。
- 之前 H2/H3 没遇到该问题，可能是当时服务已由外部启动或环境/缓存中已有 tokenizer；本轮从干净服务重启路径开始，因此暴露了 OpenCV 动态库、tokenizer cache 和 msgpack 序列化这些服务端初始化/诊断返回问题。

## 2026-07-22（下午）：H5 确认性阶段

- 按 council 意见锁定确认性协议（commit 5052501）：E1 = entropy 跨阶段（21 chunk，5 stage），E2 = benefit 标签可学性上界；预测在结果前写入 protocol.md。
- E2 完成（`results/benefit_rigor/`）：oracle_current_error AUROC 0.76-0.85（CI 不含 0.5）→ 标签可学；raw `k8_dit_flow_delta_joint7_rms` AUROC 0.685 [0.57,0.80] → flow 收敛速度有真实弱信号；但 benefit 均值为负、正例仅 13%、连续 Spearman 全横跨 0 → 收益天花板低。
- 交叉验证：该信号只在 joint7 维存在（all32 稀释后 0.515），与 `k8_prefix_joint_delta_mean` AUROC 完全相同（0.685）——flow delta 和 prefix 收敛量是同一信号的两种测量。
- E1 第一次运行 OOM：entropy 探针把全 query×全 KV 分数矩阵一次性物化，chunk 越靠后 KV 越长（约 1.65 GiB 单次分配）。修复：只对 action-register 查询计算 + 按 8 head 分块累积。此修复改变语义（不再包含 image query），与首次 3-chunk pilot 的 0.620 不可直接对比，将全部重跑 21 chunk。
- 后台运行脚本需显式 `PYTHONPATH=repo根`，否则 `eval_utils` 不可导入。
- E1 完成（`results/attention_entropy_stages_21/`）：21/21 chunk 采集成功，每 chunk 640 条 record（16 DiT call × 40 层）。按阶段 entropy 几乎平坦（0.5526~0.5551）；fine−free 差 +0.00095，CI [−0.00148, +0.00373] 含 0；与 benefit/motion 的所有相关 p>0.44；无 KV 长度混杂（rho=0.02）。**H5a、H5b 均按预注册规则判 no-go。**
- 三个 episode 内部 fine−free 方向一致为正但效应量仅为 chunk 内离散度的 1/50——方向或许真实但完全无实用区分度。

## 最终决策

| 子命题 | 结论 | 后续 |
|---|---|---|
| 动作序列能否分出自由/精细阶段 | 支持 | 可作为轨迹分析或执行 chunk 自适应的独立课题 |
| 参数越大、变化越快是否越难 | 否定 | 使用减速、夹爪事件和 transition 的联合描述，不用单阈值 |
| 精细阶段是否值得更多 DiT | 否定于当前离线指标 | 不实现 phase-aware more-compute gate |
| action-flow 是否能预测继续计算收益 | 当前不支持 | 闭环验证前不实现 gate |
| 固定低预算是否可能更合适 | 值得闭环验证 | 比较 2/4/8-call 成功率、稳定性和安全性 |

