# YCB 水壶和马克杯的 GraspGen 候选抓取

## 把手孔内一指、外侧一指（2026-10-19，纠正旋转轴理解后）

目标是物体保持直立，一根手指进入把手孔，另一根在把手外侧，夹住外缘实体。
此前绕夹爪局部 Z 轴的试验验证的是另一种姿态，不能用于否定这个目标。
本次通过把手截面构造候选，闭合方向具有径向分量，接近方向从把手平面正面
向外斜转 0–45°；没有继承 GraspGen 分数。几何同时检查视觉表面、cooked 表面、
台面、手腕、接近段和内外两面的接触点。

`runs/ycb-handle-inner-outer-v1/` 保存第一轮：水壶正对把手平面的候选均碰撞；
马克杯 3 个候选的 12 次测试虽能抬升，但因禁止碰撞全部不通过。
`runs/ycb-handle-inner-outer-v2/` 保存浅插入／斜向接近：水壶选 12 个候选，
36 次终点测试有 1 次成功；马克杯选 12 个候选，38 次测试有 1 次成功。
成功候选分别为 `pitcher-innerouter-oblique-007` 和 `mug-innerouter-oblique-011`。

同一成功条件的既有 17 例局部复测规则：5 次名义重复至少 4 次成功，物体 XY
±5 mm 的 8 个扰动全部成功，底盘 XY ±5 mm 的 4 个扰动至少 3 次成功。
原严格标准下水壶为 **1/5、6/8、3/4**，未通过；马克杯为 **4/5、8/8、4/4**，
通过。结果和保留列表在 `v2/repeatability/<object>/review.json` 与
`retained-references.json`。仍未测试专家完整接近路径；没有新增专家轨迹或训练。

为满足孔内插指，增加可选 `controller.gripper_open_width_m`：默认仍为 0.08 m，
本次水壶使用 0.04 m，马克杯使用 0.02 m；初始化、接近／停稳控制、静态全机器人
几何和候选筛选均使用同一宽度。原始资产、质量、摩擦、40 个默认参考和成功阈值
均保留。动作目标检查覆盖 80／40／30／20 mm 的张开、闭合与重置，物理轨迹也记录
了实际宽度。新的姿态不能在未声明预张开宽度的情况下直接加入旧参考库。

用户提出可以接受手腕辅助接触后，新增 `validate_rl.py --audit-wrist-contact`：
记录手腕/掌部 `right_fr3_link7` 对各目标的力，并在同一轨迹上额外计算只允许其
接触目标物体的假设判定。原始 `passed` 字段仍按严格规则；台面、地面、自碰撞、
其他机器人部位接触目标以及抬升、双指接触、滑移、保持时间要求均不放宽。
这是独立判定对照，不是修改默认训练规则。

## 把手抓取绕夹爪自身 Z 轴旋转 90° 的局部尝试（2026-10-19）

结果保存在 `runs/ycb-handle-roll90-v1/review.json`。从严格专家测试成功的
2 个马克杯把手参考、18 个水壶把手参考出发，保持接近轴不变，绕夹爪自身 Z
轴旋转 ±90°，在局部坐标平移网格内检查 300／2700 个姿态。新姿态不继承
GraspGen 分数或物理成功证据。视觉与 cooked 表面、张开手腕、8 cm 接近段、
台面和双指首次接触筛选后，测试马克杯 2 个姿态、水壶 3 个姿态。

终点物理测试：马克杯 **0/4**；水壶 **0/8**。再对水壶 3 个姿态沿闭合轴
移动约 11–12 mm，使两指接近同时接触，补测仍为 **0/8**。马克杯同类居中
调整会使手腕低于台面约 8.4–8.6 mm，因此未送入仿真。碰撞合格的水壶测试
没有达到 8 cm 抬升门槛。此结果只适用于本次小范围搜索，不能证明该抓取方向
普遍不可行；测试直接初始化在抓取终点，不涉及 Giskard 接近路径。

脚本：`try_handle_roll.py`、`center_handle_roll.py`、`preview_handle_roll.py`。
`validate_rl.py --placement-cases` 支持指定物体放置位置并保留正常的 54 组
底盘／手臂 IK 搜索，与指定终点种子的 `--cases` 互斥。几何对比图片在两个
物体目录的 `roll90-comparison-{front,reverse}.png`；左侧原参考，右侧旋转候选，
手指位置为独立首次接触的几何示意，不是成功保持截图。没有修改默认 40 个参考，
没有新增专家轨迹或启动训练。

## 专家到达精度对照（2026-10-19）

用户授权后，在 `runs/ycb-expert-precision-v1/` 对两个物体各 40 个参考做一次
配对比较：每参考固定同一个初始场景，原设置／严格设置各执行一次，两个物体共
160 次完整流程。场景选自上轮 400 例预检，每参考取第一个案例；没有按成功与否选择。
物体、摩擦、参考、初始状态、动作接口和物理成功标准不变。

严格组仅调整到达及超时设置：交接精度从 15 mm／0.15 rad 收紧为 5 mm／0.05 rad，
Giskard 停止阈值从 8 mm／0.08 rad 收紧为 2 mm／0.02 rad；30 秒仍未对准就以
`approach_timeout` 结束，保持接近阶段和张开的夹爪。默认配置仍保留原行为，
新行为通过 `controller.close_on_approach_timeout: false` 和顶层 `expert_precision`
显式启用。新增闭爪开始时的误差记录用于观察交接后的偏移，不改变执行动作。

结果：马克杯成功 **19→22/40**（7 个改善、4 个退步）；水壶 **14→18/40**
（5 个改善、1 个退步）。闭爪开始时位置误差中位数分别 **7.49→2.36 mm**、
**7.75→2.28 mm**。严格组仍有碰撞（马克杯 6、水壶 13），且闭爪开始时仍有
2/30、5/31 个案例超出 5 mm 或 0.05 rad，说明通过交接门槛不保证停稳后仍在门槛内。
严格组超时 5／1 个案例均验证保持 phase=0，所有正常交接均满足严格阈值。

这是精度和超时处理的组合干预，每参考只有一个场景；不能解释为稳定成功率或
把所有改善单独归因于某一个参数。候选配置独立保留，默认任务没有切换，没有
启动正式配额收集或训练。`comparison.json`、各物体的 `paired-results.csv` 和
`*-data-audit.json` 保存结果、逐参考变化与数据审计。
`report_precision_comparison.py` 检查代码／配置哈希、两组初始状态一致、严格交接
误差与超时行为，并生成汇总；`plan.json` 和 `source/` 保存复现输入。

`preview_strict_successes.py` 将严格组成功案例最后一帧的相对目标位姿和手指关节
观测还原为实际保持姿态，生成各物体 `strict-success-preview/` 中的正面／背面
PNG 与逐姿态来源 `manifest.json`。用独立的物体原点相对观测检查坐标还原误差，
并校验轨迹哈希和夹爪开口。图中为测量姿态的网格重建，不是仿真截图或目标参考
姿态；马克杯显示 22 个（2 个把手），水壶 18 个（全部把手）。

## 专家轨迹收集（2026-10-18 已获用户确认）

用户已要求按碗的流程收集两个物体的专家轨迹。当前执行目录为
`runs/ycb-expert-collection-v1/`，输入分别冻结自 mug v4 与 pitcher v5。
此前各节的“等待确认”是历史阶段状态，本次授权已覆盖轨迹生成，未授权训练。
先执行每参考一个标称完整接近测试，再做每参考十个、共 400 个完整流程案例。
每个参考至少一次物理成功才进入批量收集；端点 IK 抓取成功不替代这个门槛。

`collect_ycb_experts.py` 复用碗的收集器、DART 接口检查和数据审计；普通示范
每参考 250/20 条训练/验证，DART 每参考 50/5 条，合计每物体 13,000 条。
成功标准、碰撞要求与动作标签对齐检查不变。预检使用原已验证局部窗口，
禁止以碰撞失败、超时或抬升失败的案例填充配额；失败门槛会停止后续收集。
该脚本不调用 BC、PPO 或 DAgger 训练。运行状态及数量应以本轮实际结果文件为准，
计划配额不表示已收集完成。

本轮完整预检已结束：标称各 40 例，马克杯成功 17、水壶成功 9；随后每参考
10 例、各 400 例中，马克杯成功 **184 条，覆盖 26/40 参考**，水壶成功
**103 条，覆盖 18/40 参考**。马克杯另有碰撞 74、尝试失败 142；水壶碰撞
174、掉落 22、尝试失败 101。原 40 参考库仍保留，这些数字描述完整接近流程，
不是推翻端点初始化验证。两个物体均未通过“每个参考至少一次完整成功”的门槛，
因此没有启动 DART 接口配额或正式 13,000 条收集，也没有训练。
成功轨迹已保存于各 `pilot/trajectories/` 和 `expert-check-400/trajectories/`；
预检数据独立于正式配额。逐参考结果见 `reference-approach-results.csv`，
汇总见 `approach-review.json`，数据审计见各 `preflight-data-audit.json`。
下一步应处理完整接近路径和实际到达抓取终点的稳定性，再复测未通过参考；
不放宽碰撞、滑移或保持成功门槛来填满数量。

## 水壶摩擦调高后的配置（v5）

**水壶已完成 40 个参考的准备，等待专家轨迹确认。** 本轮 148 个不同姿态完成 2,516 次完整复测，53 个姿态通过（旧版 28 个中 22 个通过，两组补充候选分别新增 20 和 11 个）；按位姿最远点选择保留 **40 个，其中 35 个把手、5 个其他位置**。全部使用水壶摩擦 2.0、夹指 0.5，并通过同一 ±5 mm 复测规则。最终输入位于 `resources/validated-ycb-v5/pitcher/`，`configs/pitcher-friction-v5.yaml` 已切换到这一组 40 个输入。40 个参考对应 4 个局部放置窗口；源配置和资源目录中的配置各通过 800 次 reset／restore，40 个全部采样到，120 个审核初始状态和 40 个评估初始状态均可恢复。没有生成专家轨迹或训练。

`retained-grasps.png`、`coverage.png` 是普通预览图片；`reference-validation.csv` 列出逐参考通过次数，`references.json` 为最终 40 个，`backup-validated-references.json` 保留全部 53 个。最小归一化位姿间距约 0.276（尺度 3 cm／30°，含双指交换对称），覆盖度仅针对本轮验证通过的有限候选集合。额外 120 个备用初筛中有 37 个单次成功，它们没有完整复测，未混入这 53 个或最终 40 个。`pilot-cases.json` 只准备了每个参考一个、共 40 个已检查的初始状态；`pilot-plan.json` 明确等待用户确认，接近路径及实际到达端点尚未验证。

用户已授权按新参数完成水壶 40 参考的复验和补齐，专家轨迹仍须另行确认。`pitcher_friction_validation.py` 的本轮记录位于 `runs/validated-reference-bank-v5/`：首先冻结旧版 28 个参考及另 120 个候选，每个条件完整执行 5 次原位、8 个物体 ±5 mm、4 个底座 ±5 mm 测试，原通过门槛 4/5、8/8、3/4 不变。来源姿态、几何检查和旧结果只用于选择测试条件；通过标签只来自本轮新摩擦下的物理结果。旧版仅 ±2.5 mm 通过的参考也重新测试 ±5 mm，不降低范围补数。`candidate-seeds.json` 是待测来源，不是可用抓取库；当前确认通过数量由该目录的 `summary.json` 给出。

按用户要求提高机器人与水壶之间相对于碗的摩擦。物理资产为 `assets/ycb/019_pitcher_base/contact-v5-friction/rl.usda`；最初的 160 候选输入仍归档于 `resources/pitcher-friction-v5/`，与上面的最终 40 个区分。水壶静／动摩擦从 1.0 改为 2.0，机器人保持 0.5，默认 average 组合下接触摩擦由 0.75 提至 1.25。水壶与台面的组合摩擦也由 1.0 提至 1.5；这是修改物体材质同时产生的变化，不能把结果称为只改变夹指接触对的实验。这些值是用户授权的仿真假设，未经实物标定。

`prepare_pitcher_friction.py` 只替换 USD 内两项摩擦标量；几何、质量和其他物理形状属性逐字保持。它复用同一 cooked 几何、静态位置过滤结果，重新绑定资产／bank／mask 哈希。原 v3／v4 资产和证据、碗与马克杯配置保留。最初的 160 个候选和旧版 28 个通过参考都未自动继承新参数下的通过标签；最终通过者均来自本轮独立的完整复测。

新参数下运行时材质读回和固定 80 案例的比较放在 `runs/pitcher-friction-v5/`。80 案例沿用先前 21 个参考的固定姿态，用于材料对照，不是对主配置 160 候选的完整验证。复现只执行诊断：

对照完成：原摩擦通过 **19/80**，新摩擦通过 **30/80**；禁止碰撞由 17 降为 13，抬升／保持失败由 44 降为 37。`comparison.json` 保存配置差异核对及结果哈希。运行时确认全部 100 个水壶形状为摩擦 2.0，机器人仍为 0.5。原成功定义保持，旧验证标签未迁移。

```bash
bash scripts/rl.sh /workspace/graspgen/validate_rl.py \
  --config /workspace/runs/pitcher-friction-v5/paired-inputs/task.yaml \
  --out /workspace/runs/pitcher-friction-v5/paired-physics-rerun \
  --num_envs 80 --headless \
  --cases /workspace/runs/pitcher-failure-diagnosis-v4/cases.json \
  --object-collision-mesh /workspace/runs/pitcher-supplement-v3/019_pitcher_base/cooked-boundary.obj \
  --trace-phases --compare-grasp-center-slip
```

## 上一版记录：马克杯 40 个，水壶旧摩擦下 28 个（v4）

**马克杯已完成：** 累计 74 个不同参考通过原标准的局部复测（16 个把手），从中按位姿覆盖选出 **40 个，其中 10 个把手、30 个其他位置**。最终输入位于 `resources/validated-ycb-v4/mug/`：`references.json` 为 40 个参考，`bank.json`、`position-mask.json`、`task.yaml` 为匹配的可执行输入，`retained-grasps.png` 为普通图片。74 个通过参考另存为 `backup-validated-references.json`。最终选择的最小归一化位姿间距约 0.501（度量尺度 3 cm／30°，考虑双指交换）；覆盖仅指已验证的有限候选集合。最终 40 个对应 48 个成功条件、4 个 ±5 mm 局部窗口，发布前后配置均通过各 800 次 reset／restore，40 个参考全部采样到。资源输入路径不依赖 `runs/`；完整历史物理证据仍保留于 `runs/`。没有生成专家轨迹或训练。

上一轮的 14 个马克杯／1 个水壶只是已通过的子集，**不表示已完成各 40 个参考的要求**。v4 保留全部旧证据，继续补测，且不启动专家轨迹或训练。实际计数以 `runs/validated-reference-bank-v4/summary.json` 为准；同一个姿态的多个成功放置条件不重复计数，跨候选池相同姿态（含双指交换对称）也去重。旧水壶的 ±2.5 mm 条件单列，不冒充 ±5 mm 验证。

水壶的候选数量、物理成功案例数、通过复测的不同参考数须分开理解。例如 v4g 主复测 816 个案例成功 519 个，但 48 个受测参考仅 9 个通过全部门槛；39 个未满足物体八方向扰动全部成功（另有重叠的原点／底座失败）。这些经过筛选且相关的模拟案例不能当作 GraspGen 的通用成功率，也不足以单独归因于模型能力。`prepare_additional_conditions.py` 只补测先前未测试的成功摆放／底座条件，排除已失败的复测条件及已保留参考；`run_additional_conditions.sh` 不生成专家轨迹。

`archive_validated_progress.py` 将尚未补足 40 个的通过姿态、来源证据及每个参考自己的扰动范围保存到 `resources/validated-ycb-v4/pitcher-progress/`。此目录明确标为未完成，提供普通抓取预览和覆盖图，不提供混用不同验证范围的可执行任务配置。

v4a／v4c／v4g 的剩余成功条件补测完成后，水壶累计保留 **28 个不同姿态（23 个把手）**，其中 27 个通过物体 ±5 mm、1 个仅通过 ±2.5 mm，底座扰动均为 ±5 mm。距离统一 ±5 mm 的 40 个参考仍缺 13 个，不能将这份进度库标记为完整输入。

`audit_runtime_materials.py` 只加载环境并读取材质，不执行动作。实际 PhysX 读回显示：机器人 77 个碰撞形状均为静／动摩擦 0.5，水壶 100 个形状均为 1.0；夹指没有独立材质绑定，使用场景默认的 0.5，默认组合规则为 average。之前“摩擦=1”的描述仅对应物体，不是完整的接触对参数；这些参数未按实物标定。完整读回保存在 `runs/pitcher-failure-diagnosis-v4/runtime-materials.json`。新增 `--diagnostic-finger-friction` 仅改两根夹指的 8 个碰撞形状，记录原值和实际读回；诊断结果带 override，不进入默认配置的通过库。

同一组 80 个案例（21 个不同参考）的摩擦单变量对照已完成：夹指静／动摩擦分别为 0.5、1.0、2.0，水壶固定 1.0，原规则成功数分别为 **19、20、35**；禁止碰撞分别为 17、21、17。仅作诊断的夹持点位移规则对应 53、51、58。默认 0.5 的结果复用此前相同配置／案例／种子的完整测试，相关哈希逐项核对。提高摩擦会影响结果，但 1.0 几乎无改善；2.0 未经实物标定，不因测试更好而采纳。比较保存在 `runs/pitcher-failure-diagnosis-v4/friction-comparison.json`，不改变默认材质、成功定义或已验证库。

`expand_reference_validation.py` 按评分与位姿覆盖交替选择未测候选，重新生成匹配参考数的位置表；`run_expanded_validation.sh` 只执行物理诊断、冻结复测案例及审查。首轮优先复测每个参考保持时间最长的成功条件，其余成功条件及全部失败证据保留。每个复测条件仍是 5 次原点、8 个物体 ±5 mm 扰动、4 个底盘 ±5 mm 扰动，门槛保持 4/5、8/8、3/4。文件名 `selected-40.json` 为兼容旧消费者保留，扩展候选批次可能超过 40，不能凭文件名当作最终可用库。

水壶失败重放记录在 `runs/pitcher-failure-diagnosis-v4/`。80 个案例中，44 个为抬升／保持失败，其中 41 个在整个保持阶段超过原有 2 cm 滑移限制，32 个在整个保持阶段仍有双指接触和足够高度。`recenter_pitcher_candidates.py` 尝试沿夹爪闭合轴平移，使两指首次接触行程更接近；它是对已有 GraspGen 姿态的几何调整，不是新网络推理。原姿态、来源索引和平移量保留，调整后重新检查原网格与 cooked 几何，仍须通过相同物理验证。

`collect_validated_references.py` 汇总带来源哈希的通过姿态和条件；`finalize_reference_bank.py` 仅在 ±5 mm 下有足够通过姿态时才生成独立 40 参考输入，马克杯要求其中 10 个为把手抓取。它不允许用未通过或仅小窗口通过的候选补数，也不调用专家规划器。最终输入仍需 reset／restore 检查，专家接近路径的验证须待用户确认后另行进行。

`prepare_pitcher_neighborhoods.py` 从三个已经筛选的水壶候选池中，排除已测来源索引，围绕通过复测的姿态选择不同候选；已知成功只提供底座／机械臂 IK 初值，不继承成功标签。可选择两个底座位置，仍须逐姿态重新物理验证和局部复测。每批使用自身目录下不再修改的 `runner.sh` 快照，避免共享 shell 文件在运行中被编辑影响后续命令。两次早期批次的模拟正常完成，但 shell 后处理受共享文件编辑影响；已从完整且哈希校验通过的物理结果重新执行审查，恢复记录为对应的 `runner-recovery.json`。

新增诊断开关均不改变默认成功判定：`--trace-phases` 记录四个阶段末的手位姿跟踪、物体位移和接触力；`--diagnostic-finger-stiffness` 仅用于隔离参数对照，结果记录 override，导出工具拒绝将其冒充原配置成功证据。2000→8000 的刚度对照保留原单指 40 N 上限，但 80 次测试通过数由 19 降到 0，故不采用。24 种立方体旋转的数据增强仍未明显增加高评分竖向姿态；另以模型评分 0.05 探索竖向候选，几何及物理标准保持原样，60 个候选的 240 次初筛只有 1 次成功，该条件复测未通过。

`--compare-grasp-center-slip` 只记录另一个测量点的诊断结果，**不改变 passed 字段**。同一批 80 个物理案例，物体原点位移规则通过 19 个；改测 GraspGen TCP 处的位移时，53 个可满足相同连续保持条件。额外 34 个案例的末帧原点位移中位数 32.3 mm、TCP 位移中位数 2.8 mm、相对转角中位数约 12.2°。这反映出允许／不允许绕夹持点转动的任务语义差异，不等于自动新增了 34 个可用参考。比较记录与普通图片在 `runs/pitcher-failure-diagnosis-v4/metric-comparison.json`、`metric-comparison.png`。已向用户询问是否允许水壶抓起后转动；未获答复前，现有成功规则保持不变。

## 局部重复性验证与专家输入准备（grasp-repeatability-v3）

用户授权对有成功记录的抓取继续复测，并在生成专家轨迹前准备可审查的输入。`prepare_grasp_repeatability.py` 在运行前固定 `runs/grasp-repeatability-v3/criteria.json`：每个原成功条件做 5 次相同底盘／关节配置的重置复测、物体 XY 的 8 个 ±5 mm 周边点、底盘 XY 的 4 个 ±5 mm 偏移。原条件直接恢复保存的关节角；扰动条件固定指定底盘，从原关节角求解附近 IK，不重新搜索底盘。案例随机打散到不同批次及环境槽位；仿真仍是确定性系统，重复结果不是独立真实世界样本。摩擦、质量、尺寸和成功判据不变。

保留标准事先设定为：原条件至少 4/5、物体偏移 8/8、底盘偏移至少 3/4。物体与底盘分别扰动，没有声明同时处于两个极值时也能通过。马克杯有 61 个原成功条件，共 1,037 个案例；首批水壶有 4 个原成功条件，共 68 个案例。`validate_rl.py --cases` 支持这些明确的诊断输入，`--seed` 固定复测种子；未指定 `--cases` 时保留原端点搜索流程。`review_grasp_repeatability.py` 按冻结标准保留参考与条件，`preview_grasp_repeatability.py` 输出常规 PNG。

首批水壶四个条件均未通过，原条件复测分别仅 1/5、0/5、1/5、2/5 成功。为此有限补测一批此前未使用的合格候选：`supplement_pitcher.py prepare` 从同一个全新 GraspGen 候选池按评分和最小位姿间距 0.5 选择 40 个，编号 `pitcher-v3b-ref001`–`040`，结果在 `runs/pitcher-supplement-v3/`。这不是重新训练或改变物理参数。补测候选的右台面最外侧 38 格没有安全参考，因此明确将该批右侧 X 上界从 1.43 m 缩到 1.39 m，留下 950 格；原失败位置表保留。仅去掉空的外边缘条带，不允许隐藏内部孔洞。首次补测启动遇到宿主机绝对符号链接在容器内不可解析，已修正为相对链接，失败启动记录保留。只有补测中成功的条件才继续相同标准的重复性验证。

`prepare_repeatability_expert.py` 为通过的参考生成独立 `expert-inputs/`：匹配当前 USD／cooked 几何的 bank，重新编号索引的位置过滤表，1 cm 局部放置窗口，已验证端点条件，以及尚未执行的专家审查／评估初始状态。`kitchen.py` 新增显式启用的 `sampling_mode: validated_subregions`，支持这些不连续小窗口；默认双台面采样行为保留。局部窗口的角点及边中点有物理证据，完整窗口的静态几何由原连续扫掠检查覆盖，但不宣称连续区域的物理成功已穷尽验证。`check_retained_inputs.py` 只检查实际加载、重置、序列化及恢复，不调用规划器、不执行动作。

专家规划器目前不约束最终底盘／关节角等于保存的诊断条件，因此后续专家接近及实际到达的抓取仍须重新做物理验证，不能直接把端点成功当成整条路径成功。`endpoint-conditions.json` 明确记录这一限制。任何专家轨迹生成仍须用户确认；本轮只准备输入，不启动 Giskard，不生成示范，不训练。最终结果汇总由 `finalize_grasp_repeatability.py` 写入 `runs/grasp-repeatability-v3/summary.json`，原默认任务、已发布参考、旧诊断及物理资产保持原样。

**上一轮筛选：马克杯 14 个参考，其中 7 个把手参考，对应 25 个通过条件、4 个局部窗口。** 1,037 个复测案例有 815 个成功，其余为抬升／保持失败 205、禁止碰撞 10、端点碰撞 6、无双指接触 1。保留的是按条件达标的子集，不是按整体案例成功比例直接筛选。文件位于 `runs/grasp-repeatability-v3/mug/retained-references.json` 和 `expert-inputs/`。

**水壶在 ±5 mm 下仍无达标参考；仅在更小的 ±2.5 mm 窗口得到 1 个把手参考。** 补测初筛 160 个案例成功 10 个，涉及 9 个参考、10 个条件；其后 170 个重复性案例成功 75 个，但没有条件通过完整标准。015 和 021 的左侧成功条件在原点均达 5/5，因而另做一次预先固定的较小物体范围实验：物体偏移缩到 ±2.5 mm，底盘偏移仍为 ±5 mm，成功次数门槛保持 4/5、8/8、3/4。34 个案例中 28 个成功，只有 **pitcher-v3b-ref021-left-rear** 通过，分项为 **5/5、8/8、4/4**。这不能覆盖旧的 ±5 mm 范围。015 未通过，全部旧失败记录保留。水壶最终文件位于 `runs/pitcher-supplement-v3/narrow-repeatability/pitcher/retained-references.json` 和 `expert-inputs/`，放置窗口总宽仅 5 mm。

两份配置分别准备了 75／3 个尚未执行的专家审查初始状态及 25／1 个评估初始状态。普通图片为各 `expert-inputs/retained-grasps.png`（只画保留抓取）及各验证目录的 `review.png`（分项统计）。`prepare_repeatability_expert.py` 根据各轮实际冻结的物体偏移自动确定窗口宽度，不将窄范围结果提升为大范围可用。`reset-check/summary.json` 记录真实 Isaac 配置加载及重置检查；这里的初始状态不是专家轨迹，也不包含动作标签。

## 保存有成功记录的抓取（待复测）

`export_successful_grasps.py` 从两物体的最新修复版诊断中导出成功参考及其测试条件，保存到 `runs/grasp-success-review-v3/`。马克杯为 29 个参考（8 个把手参考，61 个成功案例），水壶为 4 个参考（3 个把手参考，4 个成功案例）。各物体的 `references.json` 保留原 ID 和抓取格式，`success-evidence.json` 记录对应资产哈希、所有测试结果，以及成功时的物体状态、实际 IK 底盘姿态和机械臂关节位置。原 40 个参考和默认配置保留。

这些文件标记为 `observed_success_only_not_expert_ready`，不是可直接替换任务的已认证抓取库。马克杯 9 个参考在四个位置中成功至少三次，其中 025、030 四次全成功；水壶四个参考均仅 1/4 成功。这是不同位置的诊断，不是同一条件下的重复成功率。下一步应复测成功条件、检查附近位置扰动，再绑定验证过的使用范围，生成与所保留子集及当前 cooked 资产一致的 bank／位置过滤表／规划配置。旧 `runs/ycb-rl-validation/` 下的专家预检配置不适用于这些新参考。任何专家接近轨迹生成仍须用户确认；本次只导出静态参考和诊断记录，没有生成轨迹或训练数据。

## 水壶全新候选与 40 个参考（pitcher-fresh-v3）

按用户要求补做全新候选生成、重新筛选和物理诊断。独立结果目录为 `runs/pitcher-fresh-v3/`；物理资产固定使用上一轮较粗修复版 `assets/ycb/019_pitcher_base/contact-v3-coarse/rl.usda`。原始扫描表面没有改变，因此重新采样该表面输入 GraspGen，不从凸块内部面采样，也不混入旧候选。

17 个新种子（4000–4016），每批 400 个候选，交替使用四种输入 yaw，产生 **6,800 个全新候选**。实际输入点云存于 `019_pitcher_base/input-pointclouds.npz`，模型、网格、种子和候选池哈希见 `pool-provenance.json`。用实际手腕／手掌／张开手指同时检查原扫描与修复后 cooked 凸体并集边界，保持评分 0.5、台面净空 2 mm、8 cm 接近距离、5 mm 路径采样间隔，剩余 **2,165 个**。两套表面都必须有独立双指首次接触，逐指闭合行程差不超过 2 mm，剩余 **2,004 个**。

对合格池整体执行位姿最远点选择，保留 **40 个全新参考**，编号 `pitcher-v3-ref001`–`040`。其中 **37 个**的双指接触在把手区域、3 个在其他区域；水壶没有人为设置把手配额。合格池本身包含 1,954 个把手候选、50 个其他候选。最大选中接触行程差为 **1.8594 mm**。覆盖距离使用原有 3 cm／30° 度量及双指交换对称性，最大未覆盖距离由仅取评分前 40 个的 **2.0833** 降到 **1.0838**（约 48%）；这只是有限候选池内的几何覆盖，不能解释成成功率或所有可行抓取的保证。

`prepare_fresh_pitcher.py` 负责双几何筛查、双表面接触筛查、选择、重新生成位置过滤表及放置审查；`report_fresh_pitcher.py` 生成常规 PNG 和汇总。`validate_rl.py` 新增可选 `--object-collision-mesh`，本轮端点检查同时使用原扫描和修复后 cooked 边界，运行时使用修复版 USD；原脚本默认行为保留。物理诊断仍是直接 IK 初始化后闭爪／抬升／保持，不生成 Giskard 专家接近轨迹，不运行训练。

本轮文件：`019_pitcher_base/selected-40.json`、`selected-40.npz`、`selection-report.json`、`bank.json`、`position-mask.json`、`task.yaml`。预览为 `gallery-40.png`（40 个参考）及 `coverage-40.png`（实际输入点云与抓取分布）；没有交互网页。原默认配置、已发布参考和马克杯结果保持原样，输入哈希记录在 `preserved-inputs.json`。

**本轮验证已完成，但物理表现仍差。** 新 40 个参考均有可用摆放位置，988 个放置格的完整物体扫掠检查通过，最小台面边缘余量约 1.93 cm；800 次 reset／序列化／restore 通过。160 个物理案例中只有 **4 个成功，4/40 个参考至少成功一次**：002、004、009、014 各 1/4 成功，其中 3 个属于把手组。失败为运行中禁止碰撞 69、端点碰撞 43、抬升／保持失败 42、IK 失败 2。几何双指接触与覆盖度筛选不能保证动态稳定抓取；这 40 个是完成诊断的候选，不能称为 40 个已验证可用抓取。

汇总为 `validation.json`，逐参考记录为 `reference-validation.csv`，逐案例明细为 `019_pitcher_base/physics/summary.json`。同一修复碰撞体的上一轮旧参考结果为 9/152、7/40；本轮参考、位置及静态端点检查均有变化，因此该数值只作背景，不是单变量对照或普遍成功率估计。没有切换默认配置，没有生成专家轨迹或训练数据，没有运行优化器。

复现顺序（候选及物理输出目录不能已存在）：

```bash
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  .runtime/graspgen-venv/bin/python graspgen/sample_pool.py \
  --objects 019_pitcher_base --fresh --batches 17 --seed-start 4000 --out runs/pitcher-fresh-v3
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .runtime/graspgen-venv/bin/python graspgen/prepare_fresh_pitcher.py screen
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .runtime/graspgen-venv/bin/python graspgen/prepare_fresh_pitcher.py select
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .runtime/graspgen-venv/bin/python graspgen/prepare_fresh_pitcher.py prepare
bash scripts/rl.sh /workspace/graspgen/validate_rl.py \
  --config /workspace/runs/pitcher-fresh-v3/019_pitcher_base/task.yaml \
  --object-collision-mesh /workspace/runs/pitcher-fresh-v3/019_pitcher_base/cooked-boundary.obj \
  --out /workspace/runs/pitcher-fresh-v3/019_pitcher_base/physics --num_envs 80 --headless
.runtime/graspgen-venv/bin/python graspgen/report_fresh_pitcher.py summarize
.runtime/graspgen-venv/bin/python graspgen/report_fresh_pitcher.py preview
```

## 水壶接触几何检查与修复（contact-v3）

对原水壶 40 个参考逐指闭合，比较原始扫描表面与实际 PhysX 凸体顶点重建的并集边界。上一版 `collision-v2` 有 17 个参考的单指首次接触行程差超过 2 mm，最大约 30.90 mm；13 个参考的完整张开手腕／手掌／手指在目标或 8 cm 接近检查中与物体相交。偏差主要涉及把手旁的壶身凸体过厚，不能只靠细分把手解决。

`assets/ycb/019_pitcher_base/contact-v3/rl.usda` 按原始扫描外形重新拟合：壶身使用 4 mm 高度分段，把手使用 1 mm 分段，共 217 个凸体。原始外观、点云采样网格、40 个参考、质量和摩擦不变；壶身仍为实心，不重建内腔。重心和惯量由 PhysX 根据新碰撞形状重新估计。候选凸体的并集通过闭合性与单实体、单把手孔检查，随后导出实际 cooked 形状再复核。

实际 cooked 复核后，张开／接近物体碰撞的参考由 13 个降为 0；首次接触行程最大差约 2.60 mm，95 分位约 1.93 mm。015、017、029、031 仍略超过 2 mm 筛查阈值，明确保留为待审查项。004、009、021、029 在本次独立静态闭合搜索中没有找到双指接触；这不是物理抓取失败的充分证明。该阈值是工程筛查规则，不是实物精度保证；点接触位置中位数可在接触片上跳动，因此主要比较手指关节行程，而不把点坐标差当作表面距离。

结果位于 `runs/pitcher-handle-audit/`。`contact-repair-comparison.png` 为常规对照图，`before/contact-audit.json` 和 `after/contact-audit.json` 为逐参考接触记录；新版独立测试配置为 `after/task.yaml`。完整物体在原有 988 个摆放格扫掠无家具相交，最小台面边缘余量约 1.93 cm。旧版资产和参考保留，水壶无需像马克杯那样重建采样网格或重新生成候选。专家轨迹仍待用户确认，不运行训练。

**有限对照已完成，默认配置未切换。** 1 mm 把手分段的物理通过数量明显下降，因此额外测试了一次 4 mm 把手分段；壶身仍为相同的 4 mm 构造。后者资产为 `contact-v3-coarse/`，共 100 个凸体，配置为 `runs/pitcher-handle-audit/coarse/task.yaml`。实际 cooked 几何的最大首次接触行程差约 2.54 mm，95 分位约 1.97 mm，张开／接近物体碰撞仍为 0，残留超阈值参考仍是 015、017、029、031。当前对照图片及根目录 `validation.json` 以该较粗版本为最终候选，同时保留细分版的全部记录。

| 碰撞版本 | 成功案例 | 至少成功一次的参考 |
|---|---:|---:|
| 上一版 collision-v2 | 21/152 | 11/40 |
| 修复后：把手 1 mm 分段 | 3/152 | 2/40 |
| 修复后：把手 4 mm 分段 | 9/152 | 7/40 |

两次新测试分别通过 800 次 reset／序列化／restore 及 988 个摆放格检查，初始案例与旧版完全一致。033 仍无可用摆放位置，008、039 仍仅左侧可用。较粗分段的物理结果有所回升，但未恢复到旧版水平；这不能单独确认下降由接触求解器引起，因为凸体分区、局部近似和惯量也会影响动力学。本轮在此停止继续调整，不把“与扫描表面更一致”当作“可稳定抓取”。`reference-review.csv` 保留逐参考的几何偏差和物理结果，`logs/` 保留构建及编译日志。未生成专家轨迹，未运行优化器。

## 马克杯连续采样网格与全新参考（surface-v3）

`assets/ycb/025_mug/surface-v3/repaired.obj` 是独立的修复表面；同目录 `rl.usda` 的外观网格使用同一表面，174 个物理碰撞块也同步调整。保留 1.5 mm 假设薄壁、原拟合杯底和杯沿、分段把手。把手根部分割面的顶点向杯身延伸 3 mm，各凸块绕自身质心扩大 0.2% 以消除数值接缝（最大顶点位移约 0.0793 mm），再用双精度布尔并集提取边界，去除内部重叠面。没有从原始凸块列表直接采样。

修复网格通过单连通实体、封闭性、绕序和拓扑检查（genus=1，对应把手孔），并通过 20,000 个表面点沿法向两侧的实体边界检查。杯腔和把手孔的空域探针通过。`repair.json` 记录参数、原始输入和新资产哈希。该模型是修复后的几何近似，并非重新测量的真实杯子；采样使用完整表面，也不是相机可见点云。加载 OBJ 时使用 `process=False`，保留布尔运算输出的拓扑，避免默认顶点合并容差损坏极短边。

运行目录为 `runs/mug-surface-v3/`：普通图片 `surface-and-pointcloud.png` 展示原扫描、修复表面、实际首批 GraspGen 输入点云与剖面。首次 6,800 个全新候选只有 8 个合格把手抓取，因此增加 8 个独立种子，共 **10,000 个全新候选**。未混入旧网格候选。完整实际手腕/手掌/张开手指同时对修复网格和 PhysX 导出碰撞体筛查后，剩余 4,302 个；再检查双指接触与净空，得到 12 个把手候选和 2,579 个其他候选，按位姿分散程度选出 **10+30** 个参考。评分、2 mm 台面净空和 8 cm 预抓取标准未放宽。

新编号为 `mug-v3-ref001`–`040`，前 10 个为把手组。`025_mug/selected-40.json`、`bank.json`、`position-mask.json`、`task.yaml` 属于独立版本，勿与旧 `mug-ref...` 混用。原有已发布资源、水壶及旧专家预检配置保持原样。新增脚本为 `repair_mug_surface.py`、`preview_repaired_mug.py`、`prepare_repaired_mug.py`、`audit_repaired_mug.py`、`preview_repaired_grasps.py`、`summarize_repaired_mug.py`；`sample_pool.py` 和 `rebalance_mug.py` 支持显式传入新网格与输出目录，默认仍为原流程。专家轨迹仍须用户确认，未运行训练。

本轮验证已完成：40 个参考均有可用摆放位置，完整物体在 1,040 个摆放格内连续扫掠未发现家具相交，最小台面边缘余量约 3.78 cm。通过 800 次 reset／序列化／restore 检查。160 个直接 IK 初始化后的闭爪／抬升／保持案例中 **61 个成功**，**29/40** 个参考至少成功一次，其中把手组 **8/10**（14/40 个把手案例成功）。其余为端点碰撞 22、运行中禁止碰撞 22、抬升／保持失败 55。没有生成接近轨迹。因为参考和几何同时改变，这些数字不能作为相对旧版的受控成功率提升证据。

汇总见 `runs/mug-surface-v3/validation.json`，逐案例记录见 `025_mug/physics/summary.json`，带结果的常规预览图见 `025_mug/gallery-40.png`。完整点云批次存于 `025_mug/input-pointclouds.npz`（25×2048×3，包含对应输入 yaw；每批种子和旋转见 `pool-provenance.json`）。修复网格与物理配置均保持原米制坐标和尺寸；物理碰撞仍是 PhysX 编译的凸体近似。

## 局部碰撞体改进试验（collision-v2）

独立候选资产位于 `assets/ycb/<object>/collision-v2/`，独立诊断配置与结果位于 `runs/ycb-collision-v2/`。原资产、已发布参考和专家预检配置没有切换。

- 马克杯：按扫描外壁拟合圆形轮廓，采用 1.5 mm 假设壁厚，保留原杯底和杯沿，将把手按高度分段。实际 PhysX 编译为 174 个凸体。内壁下部的扫描凸起不进入新碰撞体；这是一种薄壁近似，原外观网格仍保留扫描缺陷。
- 水壶：保留原主体碰撞体在分割面内的部分，将外侧把手按 8 mm 高度分段，实际编译为 64 个凸体。主体仍为实心，不用于验证伸入壶内的抓取。
- 质量、摩擦、参考姿态、摆放格、随机种子和控制器保持一致；重心和惯量由 PhysX 根据新形状重新估算。静态 IK 端点预筛仍使用原始表面网格，以保持与旧诊断可比；它不能单独证明新碰撞代理的净空，运行时仍检查禁止接触。
- 实际碰撞体在全部摆放格扫掠未发现家具相交；最小台面边缘余量水壶约 1.93 cm，马克杯约 3.78 cm，底部初始间隙约 1 mm。

`collision-comparison.png` 为常规剖面对照图，`geometry.json`、`input-audit.json`、`placement-and-solids.json` 记录几何和输入审查。PhysX 导出多边形三角化后的体积与再次取凸包有小幅差异（最大约 2.39%），不能把图中的三角网格当作精确接触面测量；实际动力学使用 PhysX 凸体。

本轮复测完成：马克杯成功案例 **57/160 → 52/160**，至少成功一次的参考 **28/40 → 24/40**；把手组 **9/10 → 6/10**。水壶成功案例 **23/152 → 21/152**，至少成功一次的参考 **12/40 → 11/40**，其中 033 仍无可用摆放格。两物体各完成 800 次 reset／序列化／restore，全部案例初始位姿与旧版完全一致。逐参考差异在 `reference-comparison.csv`，汇总在 `comparison.json`。这是各一轮有限场景诊断，没有多次重复估计随机波动，不能解释为普遍成功率或统计显著差异。新版没有自动替换已发布资产；本轮到此停止继续重建，专家轨迹仍待用户确认。

复现顺序：`improve_collision.py`（输出目录必须不存在）→ `export_cooked.py --asset /workspace/assets/ycb/<object>/collision-v2/rl.usda --headless` → `compare_collision.py` → `prepare_collision_comparison.py` → `validate_rl.py`。构建时可设 `OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2`。这条流程不生成专家轨迹，不运行训练。

## 最新版本：厨房诊断与把手配额（2026-10-18）

**补充碰撞体审查：** 实际 cooked 凸体的截面显示，两物体的把手孔均保留，但局部被凸分解侵占。水壶主体内部按实心处理；马克杯保留杯腔，但原始内壁网格已有明显不规则凸起，凸分解进一步加厚局部表面。因此下面成功数量只代表这版碰撞近似下的初筛，不能作为最终可用参考结论；应先审查／改善接触几何，再复验。详见 `runs/ycb-rl-validation/collision-audit/review.json` 与常规截面图片。未修改仿真资产，也未生成专家轨迹。

本次还用实际 cooked 物体完整体积对全部摆放格做连续 XY 扫掠：水壶 988 格、马克杯 1040 格均未发现物体与家具实体相交。固定 map yaw=0 时，离台面边缘的最小余量分别约 1.70 cm 和 3.77 cm；初始物体底部离台面约 1 mm。审查记录在 `collision-audit/placement-and-solids.json`。

当前发布版本在 `runs/ycb-rl-validation/`。马克杯重新选择为 **10 个双指把手参考 + 30 个其他双指接触参考**；编号 `mug-ref001`–`mug-ref010` 为把手组。筛选还加入了 RL 实际使用的 cooked 手腕碰撞形状，排除了两个手腕会碰台面的候选。原先同名 ID 的姿态已归档在 `runs/graspgen-ycb-40refs/` 和 `runs/ycb-rl-validation/previous-published/`，不要混用旧 ID 对应的图片或统计。

本次在 Isaac Sim 中使用项目原来的 GARMI、厨房、接触传感器、关节限制和 fixed-finish 控制器，直接用 IK 初始化到参考抓取位姿，执行闭爪、抬升 12 cm 和保持。成功判据沿用 RL：双指接触、离台至少 8 cm、滑移小于 2 cm、无禁止碰撞并持续保持至少 2 秒。诊断额外拒绝任何时刻出现禁止碰撞的案例。这里只初始化机器人到目标，不把物体附着到手上，也不录制专家状态／动作示范。

| 物体 | 厨房几何检查 | 有可测位置的参考 | 物理诊断次数 | 至少成功一次的参考 |
|---|---:|---:|---:|---:|
| 水壶 | 40 | 39 | 152 | 12 |
| 马克杯 | 40 | 40 | 160 | 28 |

马克杯把手组 10 个中有 9 个至少成功一次；这不等于 90% 场景成功率。每个参考最多测试左右台面、前后位置共 4 次，并搜索有限的底盘位置／朝向和机械臂 IK 初值。失败保留为碰撞、IK、接触或抬升保持失败，不能据此证明所有其他位置都不可行。水壶 `pitcher-ref033` 在位置过滤阶段即无可用位置；008、039 仅在左侧有可用位置。水壶摆放区域后沿由 map Y=7.63 m 收到 7.61 m，以保证其完整占地离台面边缘至少 1 cm。

新增 USD 保持原始尺寸，采用 PhysX 64 凸体分解，实际 cooked 几何导出在 `assets/ycb/<object>/cooked.pkl`，用于后续 Giskard，而不是复用碗的碰撞形状。质量使用 [YCB 原论文](https://personalrobotics.cs.washington.edu/publications/calli2015ycb.pdf)的水壶 178 g、马克杯 118 g；摩擦系数 1.0 沿用原碗的仿真假设，没有经过实物标定。

主要结果：

- `runs/ycb-rl-validation/validation.json`：总体结果与限制。
- `runs/ycb-rl-validation/reference-validation.csv`：全部 80 个参考的逐项结果。
- `runs/ycb-rl-validation/validation-overview.png`：静态结果图。
- 各物体目录的 `gallery-40.png`：新版 40 个姿态的常规图片。
- 各物体目录的 `physics/summary.json`：每个案例的 IK、碰撞、抬升和保持结果。
- 每个物体还通过了 800 次正常 reset／序列化／restore 检查。

**尚未生成 Giskard 接近轨迹，尚未做完整专家路径预检或训练。按照用户要求，在任何新物体专家轨迹生成前暂停确认。** 已准备 `expert-task.yaml`、各自的 bank、位置过滤表与 390/400 个专家预检初始场景；这些仅是配置与初始状态，不是示范。水壶 033 没有可用位置，因此不在这 390 个场景中；不能将它视为通过验证。专家预检将保留失败，并且只做测试，不运行训练监督脚本。

本次新增脚本：`rebalance_mug.py`、`prepare_rl_validation.py`、`validate_rl.py`、`export_cooked.py`、`prepare_expert_preflight.py`、`finalize_validation.py`。`validate_rl.py` 的输出目录必须不存在；不会启动 Giskard 或优化器。下面各节保留先前的候选生成及几何筛查记录。

以下记录最初使用官方 GraspGen 的 Franka-Panda diffusion checkpoint 生成抓取候选的流程；候选生成本身不启动 RL 训练或机器人运动。

输入为 YCB `019_pitcher_base` 和 `025_mug` 的 Google 16k 原始网格，单位米、缩放 1.0。下载地址和哈希保存在 `resources/ycb-graspgen-sources.json`。本地资产位于 `assets/ycb/<name>/google_16k/`。

## 运行

在仓库根目录执行：

```bash
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  .runtime/graspgen-venv/bin/python graspgen/generate.py \
  --out runs/graspgen-ycb
```

输出目录不能包含已有的 `summary.json`，重跑时指定新的 `--out`。默认每个物体采样 2048 个表面点，生成 400 个候选，导出评分前 100 个。随机种子为 42；GPU 算子不保证逐位确定性。

## 输出与坐标

- `summary.json`：模型/网格哈希、源码版本、参数和统计。
- `<object>/all_candidates.npz`：所有候选的 4×4 变换、模型评分、输入点云、假设桌面净空。
- `<object>/top_grasps.json`：前 100 个候选；位置单位米，四元数顺序 **wxyz**。
- `<object>/top_grasps.yaml`：上游 Isaac grasp 格式导出。
- `<object>/top6.png`：评分最高的 6 个姿态；黄色为官方 Panda 张开夹爪模型。
- `<object>/viewer.html`：自包含网页，可离线旋转、缩放、切换前 20 个候选。

`T_object_graspgen` 将 GraspGen 夹爪局部坐标映射到输入 OBJ 的原始坐标；夹爪 +Z 为接近方向，X 为闭合轴。采样点云的中心平移由上游 sampler 内部撤回，结果不在中心化坐标系中。这里没有把姿态转换成 GARMI 末端坐标，也不直接写入现有碗的参考库。

模型评分不是物理成功率。候选未经 GARMI IK、物体碰撞、手臂碰撞、接近路径、闭合或抬升验证；也没有强制只抓把手。`plane_clearance_m` 仅检查最终张开夹爪相对 `z = mesh.bounds[0, 2]` 无限水平面的最低点：负数说明在这个假设摆放下会穿过支撑面，正数不等于实际可执行。

## 环境来源

- [GraspGen](https://github.com/NVlabs/GraspGen)，源码位于 `third_party/GraspGen`；commit `2dd8852e1be60f5f9d277fafcc621835cdf59110`。
- [GraspGenModels](https://huggingface.co/adithyamurali/GraspGenModels)，revision `ec1ccbb5eec0680db669246ac312a3636f16ee43`；只下载 `graspgen_franka_panda.yml` 和对应 generator/discriminator 权重。
- 独立 Python 3.10 环境 `.runtime/graspgen-venv`；PyTorch 2.1.0+cu121；CUDA 12.1 编译 PointNet2，A40 架构 8.6。

基础安装命令（在仓库根目录）：

```bash
uv venv --python /usr/bin/python3 .runtime/graspgen-venv
uv pip install --python .runtime/graspgen-venv/bin/python \
  -e third_party/GraspGen --find-links https://data.pyg.org/whl/torch-2.1.0+cu121.html
uv pip install --python .runtime/graspgen-venv/bin/python wheel ninja plotly
CUDA_HOME=/usr/local/cuda-12.1 CC=/usr/bin/g++ CXX=/usr/bin/g++ \
  CUDAHOSTCXX=/usr/bin/g++ TORCH_CUDA_ARCH_LIST=8.6 MAX_JOBS=4 \
  uv pip install --python .runtime/graspgen-venv/bin/python \
  --no-build-isolation ./third_party/GraspGen/pointnet2_ops
```

YCB 原始网格遵循 [CC BY 4.0](https://ycb-benchmarks.s3.amazonaws.com/index.html)。GraspGen 代码和权重遵循各自上游许可。

## 首轮结果（2026-10-18）

结果位于 `runs/graspgen-ycb/`。

| 物体 | 候选数 | 导出数 | 最高模型评分 | 前 100 个中不穿过假设水平面的数量 |
|---|---:|---:|---:|---:|
| `019_pitcher_base` | 400 | 100 | 0.9571 | 96 |
| `025_mug` | 400 | 100 | 0.9521 | 100 |

从前 6 个姿态预览观察，水壶主要抓侧把手，马克杯主要从上方夹杯沿；这不代表所有候选的分类。所有原始候选均保留，未按桌面净空删除候选。

权重 SHA256 与固定 Hugging Face revision 的 LFS 哈希一致。导出检查验证了 400 个变换的形状、有限值、旋转正交性和行列式，以及前 100 个 JSON/NPZ 对应关系和 YAML 四元数往返误差（最大小于 2e-7）。环境版本记录在 `runs/graspgen-ycb/environment.txt`，检查结果在 `export-validation.json`。没有运行 RL 训练或物理抓取验证。

## 每个物体的 40 个多样化参考

2026-10-18 的选取结果：

| 物体 | 扩充候选池 | 分数与几何筛查后 | 保留 | 最坏覆盖距离：仅取高分 → 多样化选取 |
|---|---:|---:|---:|---:|
| 水壶 | 6800 | 2448 | 40 | 2.513 → 1.154（降低 54.1%） |
| 马克杯 | 6800 | 2586 | 40 | 3.293 → 1.632（降低 50.5%） |

这里的距离是下面定义的无量纲位姿距离；改善的是最坏覆盖缺口，不能解释成成功率提升。水壶的参考主要分布在把手不同高度和方向；马克杯覆盖杯沿与把手附近的不同接近姿态。

### 筛选规则

1. 保留首轮全部 400 个候选，额外用 16 个种子（1000–1015）、4 种输入 yaw（0/90/180/270°）各生成 400 个，共 6800 个。旋转仅用于模型输入增强，输出全部映回原始 OBJ 坐标。
2. 模型评分至少 0.5；采用当前 GARMI URDF 的手掌和双指碰撞形状，夹爪开口 8 cm。张开夹爪的目标位姿不与物体网格相交，且后退 8 cm 到预抓取位姿的直线路径每 5 mm 检查一次物体碰撞。整段路径离假设水平支撑面至少 2 mm。
3. 从通过筛查的候选中，先选评分最高者，再反复选取离已选参考最远的姿态，直到 40 个。并非取分数前 40，也未人为设置把手/杯沿配额。
4. 位姿距离为 `sqrt((TCP位置差/0.03m)^2 + (旋转差/30度)^2)`。TCP 使用 GARMI 手坐标中的 `[0,0,0.0934]m`；平行夹爪绕局部 Z 轴旋转 180°视为交换两指的同类姿态，避免浪费两个名额。

评分阈值是实验参数，不是经标定的成功概率阈值。选中水壶评分范围 0.502–0.965，马克杯 0.509–0.963；为了覆盖稀疏区域，部分参考评分低于最高分集合。

### 格式与文件

参考列表与 `resources/bowl-40-references.json` 使用相同字段：`id/rank/score/mesh_T_gripper/position/quaternion`。这里 `rank` 是扩充池中的全局评分排名，ID 的顺序是多样化选取顺序。

- `resources/pitcher-40-references.json`：水壶 40 个参考。
- `resources/mug-40-references.json`：马克杯 40 个参考。
- `resources/ycb-40-reference-selection.json`：参数、来源哈希、筛选统计与覆盖度对照。
- `assets/grasp_bank/ycb-selected-40.json`：两个物体的合并参考库，状态为 `geometry_screened_not_physics_validated`。
- `runs/graspgen-ycb-40refs/<YCB物体>/viewer-40.html`：离线可旋转网页，可逐个查看全部 40 个参考。
- 同目录 `coverage-40.png` 是分布图，`gallery-40.png` 是全部参考的缩略图，`selected-40.npz` 同时保存两个坐标约定下的矩阵。

**坐标约定与碗一致：** `mesh_T_gripper` 保留 GraspGen 原始变换；给 GARMI 的 `position/quaternion` 来自 `mesh_T_gripper @ Rz(-90°)`，四元数为 wxyz。已对现有碗的全部 40 条参考验证这一关系；新文件共 80 条导出也通过了矩阵/四元数检查。

当前只在生成且通过阈值/几何检查的有限候选集合内优化覆盖度，不能证明覆盖所有物理可行抓取。碰撞检查使用三角网格和 URDF 几何，不等同于 Isaac 的接触动力学；离散路径检查不保证采样点之间绝对无碰撞。尚未验证闭爪接触、抬升保持、机器人 IK、手臂/家具碰撞或厨房位置过滤表。合并库尚无新物体的 USD 和质量配置，不能直接替换碗任务的运行清单；原碗参考、任务配置和位置过滤表未改动。

### 复现

在仓库根目录，用新的输出目录运行：

```bash
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  .runtime/graspgen-venv/bin/python graspgen/sample_pool.py --out runs/graspgen-ycb-40refs
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  .runtime/graspgen-venv/bin/python graspgen/select_references.py --out runs/graspgen-ycb-40refs
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  .runtime/graspgen-venv/bin/python graspgen/preview_references.py --out runs/graspgen-ycb-40refs
.runtime/graspgen-venv/bin/python -m pytest -q tests/test_grasp_reference_diversity.py
```

`sample_pool.py` 会拒绝覆盖已有候选池。筛选脚本可以复用带输入哈希的几何缓存，并更新上述参考列表。这里的原碗库来自迁移资产；仓库没有它最初的参考选择脚本，因此本次实现了明确记录的多样性筛选规则，并复用了碗的参考格式和坐标约定，没有声称复现其未知的原始筛选算法。
