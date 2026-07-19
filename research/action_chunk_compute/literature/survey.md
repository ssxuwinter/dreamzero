# 一手文献与本研究位置

## 直接相关工作

### DreamZero / World Action Model

- 论文：[World Action Models are Zero-shot Policies](https://arxiv.org/abs/2602.15922)
- 关键信息：DreamZero 基于自回归视频扩散骨干联合预测未来世界状态和动作，并通过系统优化实现闭环控制。
- 对本研究的含义：动作不是独立的小策略头输出，而是和未来视频共同去噪；动态计算必须同时考虑 action flow 与 video flow 的收敛，不能只套用普通 VLA 的置信度门控。

### Diffusion Policy

- 论文：[Diffusion Policy: Visuomotor Policy Learning via Action Diffusion](https://arxiv.org/abs/2303.04137)
- 关键信息：动作由迭代去噪过程生成，并采用 receding-horizon control。
- 对本研究的含义：动作 chunk 的最终几何与去噪过程的收敛是两个不同信号；本研究需要测量后者是否预测额外迭代价值。

### ACT 与精细操作

- 论文：[Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware](https://arxiv.org/abs/2304.13705)
- 关键信息：精细操作的困难来自精度、接触力协调和闭环视觉反馈，动作分块用于降低长时误差累积。
- 对本研究的含义：动作速度或幅度不是物理难度的完整定义；接触与反馈需求是未被 action-only 特征直接观测的混杂因素。

### 自适应动作分块

- 论文：[Adaptive Action Chunking at Inference-time for Vision-Language-Action Models](https://arxiv.org/abs/2604.04161)
- 关键信息：使用多次动作预测得到的 action entropy 自适应决定执行多少个动作，以平衡响应性和 chunk 间不连续。
- 对本研究的含义：已有工作支持“输出动作的不确定性可驱动自适应决策”，但它调节的是执行 chunk 长度，不是一次 WAM 请求内部的 DiT 调用数，因此不能直接证明师兄的设想。

### 扩散模型早退

- 论文：[A Simple Early Exiting Framework for Accelerated Sampling in Diffusion Models](https://arxiv.org/abs/2408.05927)
- 关键信息：扩散不同时间步所需的 score-network 计算量不同，可使用随时间变化的 early-exit schedule 跳过部分网络参数。
- 对本研究的含义：自适应扩散计算本身可行，但该方法依据扩散时间步而非机器人操作阶段；师兄的想法需要额外证明阶段/动作信号与计算收益相关。

### 机器人运行时监控

- 论文：[Model-Based Runtime Monitoring with Interactive Imitation Learning](https://arxiv.org/abs/2310.17552)
- 关键信息：接触丰富任务中的策略错误监控存在困难，夹爪事件附近容易出现误报，细微视觉差异也可能造成关键后果。
- 对本研究的含义：只用夹爪变化做 gate 很可能把“事件发生”误当作“模型需要多算”，必须与真实配对收益对照。

## 研究缺口

现有工作分别证明了动作 chunk 可自适应执行、扩散网络可动态分配计算、接触阶段具有更高物理风险，但没有直接证明：**一个 WAM 已生成或正在生成的动作 chunk 的幅度/变化率，能够预测继续运行更多 DiT 对动作质量的边际收益。** 本项目的核心贡献应是验证或否定这个连接，而不是再次证明抓取阶段可以从夹爪信号识别。

