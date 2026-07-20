# 研究发现

## 最终判定

状态：**阶段识别 go；按阶段多算 no-go。**

师兄的想法包含两个命题，实验只支持第一个：

1. **支持**：可以从动作 chunk 的夹爪事件、关节减速/几何和相邻 chunk 变化中识别自由移动与精细操作阶段。
2. **不支持**：精细阶段没有表现出更高的额外 DiT 计算收益；动作几何、chunk 间变化、action-flow 收敛和 GT phase oracle 都不能可靠预测“继续多算是否更好”。

因此不能实现“自由移动少算、精细操作多算”的阶段感知 gate。阶段分类器仍可用于轨迹分析或自适应执行 chunk 长度，但那是另一个应用。

## 三层证据

### 1. GT 动作阶段，1,200 episode

- 11,030 个连续 chunk，10,674 个具有清晰阶段代理。
- `pre-close vs free`：joint-only AUROC 0.687，gripper-only 0.699，joint+gripper 0.787，加入 transition 后 0.800。
- transition 相对 joint+gripper 的独立 AUROC 增量为 0.0132，episode bootstrap 95% CI `[0.0038, 0.0225]`。
- closing 的关节平均步长、加速度和 jerk 中位数均低于 free；“越大、越快、变化越剧烈越难”不成立。

### 2. WAM 静态 5/8/16-call，12 episode x 7 chunk

- H8 q99-normalized joint MAE：5-call 0.02108，8-call 0.02121，16-call 0.02174。
- `E5-E16` 均值 -0.00066，95% CI `[-0.00167, 0.00049]`；没有总体正收益。
- WAM predicted-gripper 能识别 fine/free，episode-held-out AUROC 0.911。
- 同一批动作特征预测 `benefit>0.002` 的 AUROC 仅 0.388-0.461；GT phase oracle 为 0.327。

### 3. 同轨迹 prefix-stop，12 episode x 7 chunk

- prefix-16 与正常 full-16 逐元素一致，最大差 0，排除了不嵌套静态 mask 的混杂。
- H8 joint MAE 随调用数总体上升：1-call 0.01818，2-call 0.01817，4-call 0.01946，8-call 0.02119，16-call 0.02174。
- 各阶段 `E2-E16` 均为负：free -0.00185、pre-close -0.00286、closing -0.00340、hold -0.00420、release -0.00573。
- flow、provisional convergence、动作几何、transition、lagged previous chunk 和 GT phase 的 held-out R2 几乎全部为负；`benefit>0.002` 最佳 AUROC 约 0.609，不满足 gate 证据要求。

## 对“100 步动作”的解释

若一条轨迹有 100 步，可以把它按连续窗口分成自由移动、pre-close、closing、hold、release。前段常表现为较大的路径和速度，后段常表现为减速、平滑、夹爪切换与方向调整。相邻窗口统计量确实提供边界信息。

但是阶段边界只能说明动作语义发生变化，不能说明扩散模型继续去噪会更接近该条示范。在当前 DreamZero/DROID 离线指标下，两者没有建立关系。

## 关键边界

- 最终 chunk 输出只能调度下一 chunk；当前请求早停必须使用当前 action flow 或 provisional action。
- action-only 无法观测爪尖到物体距离、真实接触力或滑移。
- DROID 单条成功示范具有多模态问题，离线 MAE 可能偏好保守动作；不能据此宣称 1-2 call 的闭环成功率更高。
- 下一步若追求加速，应先在闭环任务成功率上比较固定 2/4/8-call，而不是实现阶段感知多算 gate。

