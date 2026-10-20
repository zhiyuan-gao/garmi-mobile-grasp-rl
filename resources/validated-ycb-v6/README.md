# 水壶和马克杯：允许手腕接触目标的验证参考

每个物体保存 40 个不同参考，均达到预设局部重复性门槛，并有至少一次严格完整接近、闭合、抬升和保持的成功记录。选择采用位姿覆盖距离，包含新的一指在把手孔内、另一指在外侧的抓法。

| 物体 | 参考数 | 把手类 | 其他 | 内外夹把手（包含在把手类中） | 数据与静态预览 |
| --- | ---: | ---: | ---: | ---: | --- |
| 水壶 | 40 | 37 | 3 | 1 | [参考](pitcher/references.json) · [正面](pitcher/40-grasps-front.png) · [背面](pitcher/40-grasps-reverse.png) |
| 马克杯 | 40 | 10 | 30 | 1 | [参考](mug/references.json) · [正面](mug/40-grasps-front.png) · [背面](mug/40-grasps-reverse.png) |

配置入口：[水壶](../../configs/pitcher-wrist-v6.yaml)、[马克杯](../../configs/mug-wrist-v6.yaml)。物体质量、摩擦和碰撞资产沿用各自已经冻结的版本。

## 验证门槛与适用范围

- 每个保留条件做 17 例局部验证：5 次名义重复至少成功 4 次；物体 XY ±5 mm 的 8 个扰动全部成功；底盘 XY ±5 mm 的 4 个扰动至少成功 3 次。
- 完整接近从正常停放机械臂姿态及合法底盘起点开始，禁止瞬移到抓取终点。交接及开始闭合时，位置误差均不超过 5 mm，角度误差均不超过 0.05 rad；超时不会强制闭爪。
- 实际抬升至少 8 cm、两指接触、脱离支撑、滑移小于 2 cm，连续保持至少 2 秒。仅放行 `right_fr3_link7` 与目标物体的接触；其他禁止接触的力阈值仍为 0.1 N。
- 发布的每个局部放置窗口都有相应局部验证和完整接近成功记录。这是已验证区域与起点下的可用参考库，不是任意随机起点或任意摆放下的成功率保证。

两套输入各通过 400 个随机重置案例及全部 40 个参考的恢复检查，逐个验证了参考索引、预张开宽度和观测中的手指位置。

## 文件

- `references.json` / `bank.json`：参考及物体资产绑定。`position`、`quaternion` 是物体坐标系下 GARMI hand 的目标位姿，单位为米、四元数为 wxyz；`mesh_T_gripper` 保留 GraspGen 坐标定义；`opening_m` 是该参考的预张开宽度。
- `position-mask.json` / `endpoint-conditions.json`：经过验证的局部放置区域和条件。
- `validation.json`：逐参考的局部复测结果、完整接近证据、代码与资产来源。
- `validation-traces/`：验证证据；每个发布窗口保留相应成功轨迹。它们不作为正式训练数据集。
- `audit-cases.json`：已经成功过的审计起点；不是留出测试集。`evaluation-cases.json` 当前为空，正式收集前仍需单独冻结训练/评估划分。
- `40-grasps-*.png` / `preview-manifest.json`：由实际成功保持状态重建的静态图片及编号映射。
- `runtime-check/summary.json`、`reference-index.csv`、`summary.json`、`SHA256.json`：加载检查、索引、完成状态及文件校验。

## 规划配置

`garmi_task.contact_policy: fingers_and_wrist` 启用目标限定的手腕接触。新配置显式采用 `expert_collision.model: usd_adjacent`，相邻体排除与实际 USD 关节连接一致，自碰撞规划缓冲为 5 mm。马克杯的 `right_fr3_link6` 对外部环境使用 3 mm 规划缓冲，以适应已验证终点约 4–6 mm 的台面间距；水壶该项保留 50 mm。手指及手腕对家具的规划缓冲为 1 mm。

规划缓冲不改变物理碰撞验收。没有 `expert_collision` 的历史配置继续使用原来的规划默认行为。所有本次运行记录、排除的中断尝试和代码快照位于 `runs/wrist-contact-v6/`；中断尝试不计入最终参考。

本目录保留抓取参考发布时的验证证据。后续已完成专家采集、BC、三轮纠偏并进行了 PPO 实验；
训练状态和迁移方式见[马克杯第 2000 轮换机续训说明](../../docs/mug-ppo-2000-handoff.md)。
