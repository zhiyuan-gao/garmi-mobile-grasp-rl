# 安装与运行

## 运行条件

- Linux、Docker、NVIDIA Container Toolkit、兼容 Isaac Sim 5.1.0 的 NVIDIA 驱动。
- 固定基础镜像内含 Isaac Sim 5.1.0 和 CRAM/Giskard Python 环境；镜像摘要见 [Dockerfile](../rl/Dockerfile)。使用前需自行接受 NVIDIA 的相关许可。
- Isaac Lab v2.3.2，固定提交 `37ddf626871758333d6ed89cf64ad702aef127d0`。
- Isaac Python 3.11、PyTorch 2.7.0、skrl 1.4.3；完整 Python 约束见 [runtime-constraints.txt](../rl/runtime-constraints.txt)。宿主监督脚本需要 Python 3 和 PyYAML。
- 原工作站的 PPO 设置是 RTX 4070 12 GB、约 64 GB 系统内存。4096 环境只代表此任务的已用配置，不是所有显卡的保证。

### 1. 准备代码与运行环境

```bash
git clone https://github.com/zhiyuan-gao/garmi-mobile-grasp-rl.git
cd garmi-mobile-grasp-rl
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements-host.txt
bash scripts/setup.sh
```

默认创建独立容器 `garmi-mobile-grasp-rl`，把本仓库挂载到 `/workspace`。可用环境变量 `GARMI_RL_CONTAINER` 改名。脚本会检查已有同名容器的挂载目录，避免误用另一个项目。

#### 本服务器：Apptainer 后端

本机 rootless Docker 无法直接使用 `--gpus all`，可使用已安装的 Apptainer：

```bash
bash scripts/setup_apptainer.sh
```

它从同一个固定摘要的 Docker 基础镜像创建本项目独立的
`.runtime/container/`，安装相同版本的 Isaac Lab 和 Python 依赖。
首次构建需要下载大型镜像，并在 `/tmp` 临时展开；可用
`GARMI_RL_BUILD_TMPDIR` 指定有足够空间且 Apptainer 能识别挂载信息的临时目录。
安装成功后 `.runtime/backend` 自动选择 Apptainer，下面的 `scripts/rl.sh`
和两个监督脚本会统一使用它。`/workspace` 仍指向当前 RL 仓库，资产无需改路径。

Apptainer 默认使用当前用户可见的第 2 张 GPU（索引 `1`）。通过
`GARMI_RL_GPU=0` 可选择另一张；运行前确认显存余量。`GARMI_RL_RUNTIME=docker`
可显式选择已准备好的 Docker 后端。运行缓存与依赖均在本项目下，
不使用或修改 VLA 的 Python 环境。

CPU 专家测试同样通过统一入口执行：

```bash
bash scripts/container_exec.sh \
  /home/jovyan/cram-vrb-lab/cognitive_robot_abstract_machine/.venv/bin/python \
  -m pytest -q /workspace/tests/test_giskard_expert.py
```

### 2. 准备经过验证的资产

**仅克隆代码不能直接训练。** 此版本依赖已经导出的机器人 USD、厨房碰撞几何、碗、40 个参考和位置过滤表；不附带从原始场景重建全部导出资产的工具。

在已有 FAME_RL 工作区的机器上运行：

```bash
python3 scripts/import_assets.py --from-workspace /path/to/FAME_RL
```

导入工具只取 [asset-sources.json](../resources/asset-sources.json) 列出的文件，先验证原始 SHA256，再调整路径；40 个参考的数值、几何和过滤表的格子不变。迁移后的 bank/mask 哈希重新计算，原始哈希保留为溯源信息。数字参考也可直接查看 [bowl-40-references.json](../resources/bowl-40-references.json)。

资产导入完成后，可以把生成的 `assets/` 和 `rl/assets/` 目录单独迁移到另一台机器的同名位置。它们被 Git 忽略，使用和分发仍遵循各资产的上游许可。资产来源为 `yxzhan/cram-vrb-lab` 的 GARMI 工作区，基准提交 `41c103214a0c89899e4c5ee28428b2f31b116307`；训练导出资产还包含本地运动/碰撞适配，因此直接下载原仓库不能替代此哈希匹配的资产包。

### 3. 选择初始化方式

**重新采集全部 40 个参考，从新网络开始：**

```bash
python3 rl/prepare_40ref_trial.py --out runs/bowl40
```

这保留同样的数据使用方法和任务设置，但不是当前实验的相同模型初始化；此新增入口尚未完整跑完训练。

**复用当前实验使用的四参考模型和专家数据：**

```bash
python3 scripts/import_bootstrap.py --from-workspace /path/to/FAME_RL
python3 rl/prepare_40ref_trial.py --out runs/bowl40 --bootstrap
```

只导入所需的初始化模型、1200 条训练示范、100 条验证示范、已有纠偏标签和 1200 个回归评估场景。它们放在 `bootstrap/`，不会上传。此模式补采其余 36 个参考，合并后也是每参考 300 条训练示范。完整运行不需要复制历史实验目录。

`prepare` 冻结配置与评估场景，不启动模拟器。输出目录必须不存在。

### 4. 检查并启动

```bash
python3 rl/run_40ref_experiment.py --out runs/bowl40 --check-only
python3 -u rl/run_40ref_experiment.py --out runs/bowl40
```

`--check-only` 是文件/配置检查；**不是物理可行性检查**。正式启动后，监督脚本先做 128 环境边界重置检查、400 个专家物理预检场景、少量扰动采集、数据审计、2 个 BC epochs 和 2 次 PPO 接口检查，成功后自动继续。任何参考在专家预检中没有成功案例会停止，具体失败保留在报告中。

GPU 阶段串行执行；专家采集使用 128 个仿真环境和 8 个 CPU Giskard 工作进程。PPO 训练进程退出后再独立评估，以释放显存。可通过 `runs/bowl40/supervisor.json` 和 `REPORT.md` 看进度；每个 PPO 阶段另有 `steps.csv`、`updates.csv` 和检查点。

`--resume` 只用于已标记 failed/interrupted 的队列；完成阶段复用，普通/扰动正式示范保留已经验证的样本，并换新种子继续。其他未完成阶段会归档后重做，未完成的单个 PPO 阶段不是逐步无损恢复。不要同时启动两份同一输出目录的监督脚本。

### 5. 评估和录像

```bash
bash scripts/rl.sh /workspace/rl/evaluate_kitchen.py \
  --headless --config /workspace/runs/bowl40/task.yaml \
  --checkpoint /workspace/runs/bowl40/ppo-0500/checkpoints/update_0500.pt \
  --num_envs 240 --out /workspace/runs/eval-500

bash scripts/rl.sh /workspace/rl/evaluate_kitchen.py \
  --headless --enable_cameras --num_envs 1 --case-index 0 \
  --config /workspace/runs/bowl40/task.yaml \
  --checkpoint /workspace/runs/bowl40/ppo-0500/checkpoints/update_0500.pt \
  --out /workspace/runs/video-case-0
```

录像放在单独阶段，避免占用正式训练的显存。检查点路径应以实际 `progress.json` 为准。

## 水壶与马克杯 v6：有限次数专家采集

`graspgen/collect_wrist_pilot.py` 只采集数据，不启动 BC、PPO 或 DART。
输入为 `resources/validated-ycb-v6/` 的两个已验证参考库，每个物体 40 个参考。
输出固定在 `runs/wrist-expert-pilot-v6/`，已有输出不覆盖、不自动追加尝试。

每个参考执行 5 次，合计每个物体 200 次。运行前固定 150 次训练与 50 次验证尝试，
每个参考在两种划分中均有案例；另冻结每个物体 80 个留出评估案例，采集不执行这些案例。
对象位置在通过验证的窗口中心 XY ±4 mm 内采样，底盘在已成功接近的起点附近 XY ±4 mm 内采样，
并限制在合法起始区域；机械臂从标准 PARK 姿态开始。该划分衡量相同抓取及局部区域内新起点的表现，
不代表未见物体、未见抓法或整个厨房范围的泛化能力。

```bash
# 仅首次准备：冻结输入、划分、代码及配置校验。
OPENBLAS_NUM_THREADS=1 .runtime/graspgen-venv/bin/python graspgen/collect_wrist_pilot.py prepare

# 两个独立物体可以使用不同 GPU；每个命令完成后自动筛选并审计数据。
OPENBLAS_NUM_THREADS=1 .runtime/graspgen-venv/bin/python graspgen/collect_wrist_pilot.py collect --object pitcher --gpu 0
OPENBLAS_NUM_THREADS=1 .runtime/graspgen-venv/bin/python graspgen/collect_wrist_pilot.py collect --object mug --gpu 1
```

读取每个物体目录下的 `status.json` 查看完成状态，`report.json` 和 `per-reference.csv`
查看全部尝试的结果。用于后续训练的是该目录顶层的 `train-index.json`、`validation-index.json`，
不是 `collection/` 下的原始索引。顶层索引只保留完整物理成功且交接和实际开始闭爪时
误差均不超过 5 mm / 0.05 rad 的轨迹；原始失败和未入选结果保留以便审计。
`data-audit.json` 检查文件哈希、观测与动作的时间对应、物理成功条件、接近阶段标签和划分无重叠。
采集次数固定为 200 次，不代表最终有 200 条成功示范，也不保证每个参考都有成功数据。

### v6 正式采集队列（仅数据）

`graspgen/collect_wrist_formal.py` 使用同样的局部起点分布，复用小批量中已审计的成功轨迹，
保留原有训练/验证划分及每物体 80 个留出评估案例。每个物体依次执行：

1. `coverage`：补齐每参考至少 1 条训练和 1 条验证成功轨迹，最多 8 批。
2. `dart-check`：每参考 2 条训练、1 条验证的真实运动扰动成功轨迹，最多 12 批。
3. `expert-plain`：每参考 250 条普通训练、20 条普通验证轨迹，含可复用的先前轨迹。
4. `expert-dart`：每参考 50 条扰动训练、5 条扰动验证轨迹，含扰动检查阶段的数据。
5. 合并并审计：每物体 12,000 条训练、1,000 条验证；两个物体共 26,000 条。

配额只接受完整物理成功且交接、实际开始闭爪均满足 5 mm / 0.05 rad 门槛的轨迹。
扰动轨迹还必须实际执行过扰动，扰动动作不进入模仿标签。采集按剩余配额选择参考和划分，
随后采样新的合法起点；不会把已经执行的失败案例改到其他划分，也不重复已使用或留出的起点。
手腕仅允许接触目标物体，其他物理成功和禁止碰撞条件沿用 v6。

```bash
# 仅首次准备，已有输出目录不可覆盖。
OPENBLAS_NUM_THREADS=1 .runtime/graspgen-venv/bin/python graspgen/collect_wrist_formal.py prepare

# 每个物体独立监督，门槛通过后自动进入下一阶段；不会启动训练。
OPENBLAS_NUM_THREADS=1 .runtime/graspgen-venv/bin/python graspgen/collect_wrist_formal.py run --object pitcher --gpu 0
OPENBLAS_NUM_THREADS=1 .runtime/graspgen-venv/bin/python graspgen/collect_wrist_formal.py run --object mug --gpu 1
```

输出为 `runs/wrist-expert-collection-v6/`。每物体的 `supervisor.json` 记录当前阶段和状态；
其 `active_summary` 指向实时批次结果，`supervisor.log` 和各阶段 `.log` 保留运行输出。
进程使用独占锁，拒绝重复启动同一物体。任何阶段失败或达到尝试上限仍未满足配额时，
仅停止该物体的后续队列，保留成功轨迹并尝试审计可用的部分数据。
此时应检查失败原因，不能将 `stopped` 或 `failed` 当成完整数据集。
最终数据入口为每物体顶层 `train-index.json`、`validation-index.json`，仅在四个阶段全部通过后生成。


本次并行扩容使用独立版本 `runs/wrist-expert-collection-v6/parallel-resize-v1/plan.json`，
普通和扰动正式采集均为每物体 256 个环境、12 个 Giskard 工作进程，两张 GPU 各运行一个物体。
原计划和代码快照保留；恢复命令需指定 `--resume --plan <版本计划路径>`，并使用版本中的新种子。
`--allow-parallel-resize` 仅允许严格局部采集在显式恢复时改变并行数；配置、采样分布、配额、
已有轨迹哈希和原有划分仍须通过一致性检查。实时生效版本见 `supervisor.json` 的 `active_plan`。
每个进程新完成批次的耗时记录在 `summary.json` 的 `batch_timings_seconds`，用于实际吞吐比较。

扩容切换的原始批次、索引、哈希验证和恢复记录保存在版本目录的各物体子目录。
本次 Isaac 的 SIGINT 直接退出，中断了部分索引更新；已依据完整批次结果恢复遗漏的索引项，
轨迹文件未改写，并使用数据审计复核。旧的 `checkpoint_wrist_resize.py` 假定会收到 Python
`KeyboardInterrupt`，不应直接复用于下一次切换；下一次应使用采集循环内的完整保存退出机制。

### 水壶与马克杯：BC 和三轮专家纠偏

`graspgen/train_wrist_imitation.py` 从已完成并审计的
`runs/wrist-expert-collection-v6/{pitcher,mug}/` 读取最终训练/验证索引，
每个物体独立训练。新输出为 `runs/wrist-imitation-v6/`，不会重采或覆盖专家轨迹。
仅包括模仿训练和评估，队列没有 PPO 入口。

```bash
# 一次性冻结数据入口、代码、起点划分和训练参数；已有目录不能覆盖。
OPENBLAS_NUM_THREADS=1 .runtime/graspgen-venv/bin/python graspgen/train_wrist_imitation.py prepare
# 对两个物体各执行端到端检查：两轮小样本 BC、策略状态查询、专家标注、一次纠偏 BC、严格评估。
OPENBLAS_NUM_THREADS=1 .runtime/graspgen-venv/bin/python graspgen/train_wrist_imitation.py run --object pitcher --gpu 0 --phase smoke
OPENBLAS_NUM_THREADS=1 .runtime/graspgen-venv/bin/python graspgen/train_wrist_imitation.py run --object mug --gpu 1 --phase smoke
# 只有对应 smoke-report.json 通过才启动正式流程。
OPENBLAS_NUM_THREADS=1 .runtime/graspgen-venv/bin/python graspgen/train_wrist_imitation.py run --object pitcher --gpu 0 --phase formal
OPENBLAS_NUM_THREADS=1 .runtime/graspgen-venv/bin/python graspgen/train_wrist_imitation.py run --object mug --gpu 1 --phase formal
```

正式初始 BC 从各自的新网络开始（小测模型不复用）：100 epochs，动作 MSE，batch 1024，
学习率 0.0003，按参考均匀抽样。状态标准化只拟合训练轨迹；验证及留出评估不参与。
每轮纠偏从每参考抽取 50 个训练起点、10 个验证起点，总计 2,000 / 400 个，
以 200 环境分批运行策略。步数 5–600 间以 0.5 概率查询 Giskard，专家动作只作为标签，
不替代策略执行；失败策略回合可贡献有效查询，专家规划失败会记录并丢弃。
三轮分别累积对应划分的标签，接续上轮模型再训练 100 epochs。
纠偏专家显式继承该物体的 manifest、精度、手腕接触规则和自碰撞配置。

每轮使用原先冻结的 80 个独立留出案例作诊断评估，不生成标签，不据此选择模型；
沿用仓库的最终 epoch 检查点衔接规则。评估通过 `--require-strict-success` 检查实际闭爪起点
及交接的 5 mm / 0.05 rad 门槛，再加原物理成功条件；物体名称从案例读取。
这些结果只代表当前局部起点分布。

`supervisor.json` 记录当前阶段；BC 的 `progress.json`/`epochs.jsonl` 记录 epoch 和损失；
`evaluations.json` 保存正式初始 BC 和各轮纠偏后的成功率。
最后的 `result.json` 指向第三轮模型及累计纠偏标签数量。
每物体使用独占锁，发生错误时保留数据并停止后续阶段。修复外部问题后可带 `--resume`
恢复失败队列：完成阶段需核对命令、输入和输出哈希；未完成阶段归档后重跑，
不会声称 BC 优化器可逐步无损恢复。

### 水壶与马克杯：局部 PPO＋模仿损失

`graspgen/train_wrist_ppo.py` 读取 `runs/wrist-imitation-v6/` 的第三轮最终模型和
累计纠偏训练索引，在 `runs/wrist-ppo-v6/` 建立独立队列。每个物体目标 8,000 次更新，
rollout 24 步，专家动作 MSE 系数 3、专家 batch 1024、按参考均匀抽样、数据放 CPU；
沿用原 PPO 学习率等参数，并冻结 BC 状态标准化。

训练重置只从该物体原有 12,000 个审计通过的训练起点均匀抽样（每参考 300 个），
不使用验证集或留出评估起点；这是离散的局部训练起点池，不宣称整个厨房随机化。
`garmi_task.require_strict_success` 显式启用交接及实际闭爪误差 <=5 mm / 0.05 rad，
与原物理接触、抬升、保持、滑移、禁止碰撞条件共同决定成功奖励和成功终止。
这些都是新增配置项，旧任务默认行为不变。

```bash
OPENBLAS_NUM_THREADS=1 .runtime/graspgen-venv/bin/python graspgen/train_wrist_ppo.py prepare
OPENBLAS_NUM_THREADS=1 .runtime/graspgen-venv/bin/python graspgen/train_wrist_ppo.py run --object pitcher --gpu 0
OPENBLAS_NUM_THREADS=1 .runtime/graspgen-venv/bin/python graspgen/train_wrist_ppo.py run --object mug --gpu 1
```

队列先验证整批与部分重置，再从 1,024 环境起步，按资源测量考虑升到 2,048 / 4,096；
4,096 环境通过且资源余量允许时，再尝试 8,192。
各段累计目标更新数为 2、4、8、16，逐段恢复优化器、调度器和随机数状态。
保留至少 4 GiB 估算显存余量、下一次扩容前至少 32 GiB 可用系统内存；
8,192 档根据前三档测到的最大单位环境显存增量外推，保留程序固定开销；
短测失败会停止该物体后续队列，具体实际环境数见 `capacity-report.json`。
短测的实际更新数计入 8,000 次预算，随后继续到 500、1,000、…、8,000。
每段训练进程退出释放显存后，用原先冻结的 80 个留出案例进行严格评估。
留出结果只报告，不用于自动选择环境数、调参或选择最终检查点；最终模型为第 8,000 次更新。

`supervisor.json` 记录队列阶段；各 `capacity-*` / `ppo-*` 目录的 `progress.json`、
`updates.csv` 记录训练进度与实际耗时，`evaluations.json` 汇总严格成功率。
每段还检查模型确实更新、优化器状态存在、恢复入口状态非空、专家模仿损失参与优化，
并逐张量检查标准化统计仍与 BC 相同。恢复会从新的仿真回合开始，不是逐物理步回放。

已失败的队列可加 `--resume` 恢复；已完成阶段先核对命令及产物哈希，未完成阶段归档重跑。
如果运行计划经审计修订，使用 `--plan <计划文件>` 指定独立版本，保留原计划及代码快照。
本次增加 8,192 档及实测内存增长估计，使用
`runs/wrist-ppo-v6/revisions/capacity-memory-v4/plan.json`；
各队列实际采用的版本记录在 `supervisor.json` 的 `active_plan`。
马克杯的 8,192 实测出现 PhysX 碰撞配对缓冲区溢出，停止后使用
`revisions/mug-4096-after-8192-v5/plan.json` 从已通过的 4,096 / 第 8 次更新恢复。
异常阶段的结果未进入后续训练，详情见 `mug/capacity-8192-rejection.json`。
