# 马克杯 PPO 第 2000 轮换机续训

代码分支：`codex/ycb-wrist-v6-ppo2000`。这条分支从更新后的 `main` 创建，
保留碗演示链接，同时收录本机水壶/马克杯几何处理、参考验证、专家采集、BC、纠偏和 PPO 代码。
训练数据和模型不存入 Git；本说明的迁移包只用于**马克杯从第 2000 轮续训**。
水壶第 1873 轮及其奖励/噪声对照的模型不包含在这个包内。

源机器已于 2026-10-21 完成第 2000 轮及检查点校验后停止，没有启动第 2500 轮。
同一批 80 个留出起点严格成功 **40/80（50%）**；其余为固定收尾失败 25、
接近超时 6、禁止碰撞 9。原 BC＋三轮纠偏基线为 51/80，当前 PPO 尚未超过该基线。

## 要复制的文件

- `mug-ppo-2000-resume.tar.gz`：第 2000 轮模型与 `.resume.pt`、原配置、固定训练/评估起点、
  机器人/厨房/物体资产、参考和放置区域的依赖文件，以及原 PPO 使用的全部专家数据。
- `manifest.json` 和 `SHA256SUMS`：传输校验和代码版本记录。
- Git 仓库的上述分支。

包内包含 12,000 条原训练示范和三轮纠偏对应的 30 个标签文件及完整索引。
PPO 的专家 BC 辅助损失继续读取这些数据，因此**只复制模型不能继续原来的训练**。
无需再复制旧的 `garmi-rl-bootstrap.tar.gz` 或旧碗资产包；当前完整资产已包含在迁移包中。
不需要搬迁整个历史 `runs/`、视频、缓存或本机 `.runtime/`。
资产仍按其上游许可使用，不随代码分支公开上传。

本次导出的压缩包为 1,200,653,334 字节（约 1.20 GB / 1.12 GiB），
包含 12,304 个数据文件，解压后约 2.32 GB。压缩包 SHA256：

```text
a0cdf3d821e72581d2be551f66380cf4d47270da9112bad58d175b3d5bd17827
```

清单记录的代码提交为 `8cd1768`；同分支后续提交仅补充文档和文件属性。
恢复校验会逐个检查数据文件以及 43 个实际运行代码文件的 SHA256。

源机器已在独立工作目录实际解压本包并完成：全部文件哈希校验、
第 2000 轮模型/优化器/归一化状态检查，以及 256 个仿真环境的 1,920 次起点恢复检查。
恢复检查覆盖全部 40 个参考，包含部分环境重置和 5 步物理推进；没有执行梯度更新。
迁移工具及相关流水线的 25 项自动测试通过。这验证了包的依赖完整性，
目的机器仍需按下文安装环境并进行检查。

`.pt` 保存策略/价值网络、标准化和优化器；配套 `.resume.pt` 保存调度器、随机数状态、
累计更新/采样数量及原训练配置。两者必须配套使用，入口使用 `--resume`，不能改成
`--init-checkpoint`。恢复时重新开始仿真回合；换硬件、驱动或并行数量不保证逐位一致。

## 新电脑恢复

在传输文件所在目录校验：

```bash
sha256sum -c SHA256SUMS
```

克隆分支，解压到仓库根目录，保留目录结构：

```bash
git clone --branch codex/ycb-wrist-v6-ppo2000 \
  https://github.com/zhiyuan-gao/garmi-mobile-grasp-rl.git
cd garmi-mobile-grasp-rl
tar -xzf /path/to/mug-ppo-2000-resume.tar.gz
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements-host.txt
python3 scripts/export_ppo_resume.py verify
```

宿主目录可以不同，容器内必须仍挂载到 `/workspace`。包内路径和数据索引原样保留，
不要批量改 JSON/YAML 中的 `/workspace`，否则会破坏输入哈希和恢复配置检查。

根据 [安装说明](usage.md) 选择一种运行环境：

```bash
# Docker + NVIDIA Container Toolkit
bash scripts/setup.sh

# 或 Apptainer（不要重复构建两种后端）
GARMI_RL_GPU=0 bash scripts/setup_apptainer.sh
```

保持项目锁定的 Isaac Sim 5.1.0、Isaac Lab 2.3.2、PyTorch 2.7.0 和 skrl 1.4.3。
更换机器时重新安装依赖，避免复制带绝对路径的虚拟环境。
参考训练使用 A40 和 4096 个环境，按用户要求目的机器续训也保持 4096。
本次未在目的机器测容量，正式长跑前先执行下面的不更新模型检查。

```bash
GARMI_RL_GPU=0 bash scripts/rl.sh /workspace/rl/check_local_ppo.py \
  --headless --config /workspace/runs/wrist-ppo-v6/mug/ppo.yaml \
  --num_envs 256 --out /workspace/runs/mug-transfer-reset-check

GARMI_RL_GPU=0 bash scripts/rl.sh /workspace/rl/audit_ppo_transfer.py \
  --bc /workspace/runs/wrist-imitation-v6/mug/round-03-bc/last.pt \
  --checkpoint /workspace/runs/wrist-ppo-v6/mug/ppo-2000/checkpoints/update_2000.pt \
  --stage /workspace/runs/wrist-ppo-v6/mug/ppo-2000 --expected-update 2000 \
  --out /workspace/runs/mug-transfer-checkpoint-audit.json
```

## 续训命令

先恢复一个 500 更新的阶段，到累计第 2500 轮，保持原来的 4096 个并行环境和优化器状态。
辅助脚本虽然支持显式调整环境数量，但下面的命令不改变数量。
训练完成并释放环境后，脚本独立执行检查点校验和
80 个留出案例的严格验证，不同时占用训练和验证场景的显存。

```bash
. .venv/bin/activate
GARMI_RL_GPU=0 GARMI_RL_NUM_ENVS=4096 GARMI_RL_FINAL_UPDATE=2500 \
  nohup bash scripts/resume_mug_ppo.sh > runs/mug-resume-launch.log 2>&1 < /dev/null &
```

`GARMI_RL_FINAL_UPDATE=8000` 则按每 500 轮训练、校验、评估顺序继续到第 8000 轮。
默认输出 `runs/mug-ppo-resumed/`；脚本拒绝覆盖已有输出目录。`status.json` 记录队列阶段，
每个 `ppo-*/progress.json` 记录真实更新进度，`evaluation-*/summary.json` 记录严格成功率。
如果中断，保留最新 `.pt` 和 `.resume.pt`，从该检查点单独恢复，不能覆盖输出目录重新开始。

不要直接启动旧 `graspgen/train_wrist_ppo.py run --resume`：它依赖原机器两物体完整计划与历史文件，
也不接受已经按用户要求暂停的队列。新的辅助脚本只使用本迁移包的依赖。

## 在源机器重新导出

等待第 2000 轮、校验和最终评估都完成并自动停止后：

```bash
python3 scripts/export_ppo_resume.py export \
  --source /path/to/source-repo --code-root /path/to/this-branch \
  --out /path/to/new-transfer-folder
```

导出工具要求停止状态和第 2000 轮检查点已验证，逐个核对数据哈希，只导出实际续训依赖。
