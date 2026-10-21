# 马克杯第 6500 次模型

[下载 GitHub Release](https://github.com/zhiyuan-gao/garmi-mobile-grasp-rl/releases/tag/mug-ppo-6500-20261022)。

模型通过 Giskard 专家示范、行为克隆、三轮专家纠偏和带模仿损失的 PPO 训练，控制 GARMI 的底盘与右臂接近马克杯，随后固定程序闭爪、提起并保持。
使用 40 个已验证抓取参考，PPO 并行环境数为 4096。本发布保存到累计第 6500 次更新；后续训练仍在进行。

## 结果

| 模型与判定 | 成功 | 成功率 |
| --- | --- | --- |
| 第 6149 次，旧姿态门槛 | 34/80 | 42.5% |
| 第 6149 次，实际抓起并保持住 | 59/80 | 73.75% |
| 第 6500 次，实际抓起并保持住 | 62/80 | 77.5% |

第 6149 次前后两行使用相同权重，只改变成功标准。后续训练和评估取消了最终成功的参考姿态门槛：物体抬高至少 8 cm、双指保持接触、没有支撑、相对滑动小于 2 cm、持续保持至少 2 秒且没有违规碰撞即可成功。掉落和越界仍判失败。闭爪触发条件继续使用位置误差 5 mm、方向误差 0.05 rad，并持续 0.1 秒；成功奖励仍为 115。

第 6500 次的 18 个失败包括碰撞 12 次、抓取未成功 5 次、接近超时 1 次。
80 个固定案例覆盖同一马克杯的 40 个已知抓取参考，每个参考两个起点，位于厨房柜台的四个局部区域；它们用于比较模型，尚不代表新物体或任意柜台位置的成功率。

完整记录：[评估](../resources/mug-ppo6500/evaluation.json)、[检查点校验](../resources/mug-ppo6500/audit.json)、[同模型新旧判定对照](../resources/mug-ppo6500/paired-baseline-6149.json)、[版本与哈希](../resources/mug-ppo6500/release.json)。

## 下载内容

- `update_6500.pt`：策略、价值网络、归一化和优化器状态。
- `update_6500.resume.pt`：配套调度器、随机数状态、累计进度和训练配置。
- `mug-ppo-6500.tar.gz`：上述模型文件、当时实际运行的完整 Git 代码快照、配置、固定训练和评估起点、评估及奖励判定变更记录。
- `manifest.json`、`SHA256SUMS`：文件、运行代码及外部输入的哈希。

模型 SHA256：

```text
b6ae2305fcbb353d88801ca3f9817b74665f34f9d8238b6fcd5a96d41e082ce2
```

实际训练代码为 `73d6da26edd0a260278aaeb93537aeda7697a760`。本整合分支另外包含杯子、牛奶盒及 CRAM＋RL 演示代码；发布包保留训练时的原代码，供评估与继续训练使用。

仿真资产和专家数据不包含在这个约 30 MB 的模型运行包中，可以从已有训练工作区或已解压的第 2000 次迁移包导入。评估需要仿真资产；保持原专家辅助损失续训时还需要原专家数据。输入文件和哈希列在 `manifest.json` 中，导入工具会逐个校验。

## 准备独立运行目录

```bash
git clone --branch codex/integrate-grasp-training-and-vrb \
  https://github.com/zhiyuan-gao/garmi-mobile-grasp-rl.git
cd garmi-mobile-grasp-rl

gh release download mug-ppo-6500-20261022 \
  --repo zhiyuan-gao/garmi-mobile-grasp-rl \
  --dir .runtime/mug-ppo6500-download
(cd .runtime/mug-ppo6500-download && sha256sum -c SHA256SUMS)

mkdir -p .runtime/mug-ppo6500
tar -xzf .runtime/mug-ppo6500-download/mug-ppo-6500.tar.gz \
  -C .runtime/mug-ppo6500
python3 scripts/package_mug_checkpoint.py verify --root .runtime/mug-ppo6500
```

从保存原输入数据的工作区导入。路径是第 2000 次迁移包解压后的仓库根目录，或原马克杯训练工作区，内部有 `assets/`、`rl/assets/`、`resources/` 和相关 `runs/` 数据。

```bash
python3 scripts/package_mug_checkpoint.py import-inputs \
  --root .runtime/mug-ppo6500 \
  --from-workspace /path/to/original-mug-workspace
python3 scripts/package_mug_checkpoint.py verify \
  --root .runtime/mug-ppo6500 --require-dependencies
```

只做评估时，可在 `import-inputs` 命令添加 `--evaluation-only`，省去专家数据复制；此时使用不带 `--require-dependencies` 的 `verify` 校验运行包。

按[安装与运行](usage.md)准备 Isaac Sim 5.1.0、Isaac Lab 2.3.2、PyTorch 2.7.0 和 skrl 1.4.3。Docker 示例：

```bash
cd .runtime/mug-ppo6500
export GARMI_RL_CONTAINER=garmi-mug-ppo6500
bash scripts/setup.sh
```

该目录在容器内挂载为 `/workspace`，保留配置中的路径。正在使用的容器和工作区可以继续运行，评估或训练需要另行安排 GPU 时间。

## 评估

下面命令使用原来的 80 个固定案例和实际抓取成功标准：

```bash
bash scripts/rl.sh /workspace/rl/evaluate_kitchen.py --headless \
  --config /workspace/runs/mug-physical-success-review/ppo.yaml \
  --checkpoint /workspace/runs/mug-ppo-physical-success/ppo-6500/checkpoints/update_6500.pt \
  --num_envs 80 --seed 12345 \
  --out /workspace/runs/mug-6500-evaluation
```

结果位于 `runs/mug-6500-evaluation/summary.json`。配置使用 `require_strict_success: false`，评估命令不添加 `--require-strict-success`。

## 从第 6500 次续训

```bash
bash scripts/rl.sh /workspace/rl/train_trial.py --headless \
  --config /workspace/runs/mug-physical-success-review/ppo.yaml \
  --resume /workspace/runs/mug-ppo-physical-success/ppo-6500/checkpoints/update_6500.pt \
  --num_envs 4096 --updates 7000 --eval-updates 7000 \
  --skip-evaluation --seed 2026102002 \
  --out /workspace/runs/mug-continued-7000
```

`--updates 7000` 表示累计到 7000，即追加 500 次；改成 8000 表示追加 1500 次。`--resume` 会恢复训练状态，同时开始新的仿真回合。新输出目录需要尚未存在。

保持 `.pt` 与 `.resume.pt` 同目录且文件名配套。训练完成后，用上面的评估命令替换 checkpoint 和输出路径即可比较。运行包中的 `runs/mug-physical-success-review/audit_physical_ppo.py` 可按原 `rl/audit_ppo_transfer.py` 的参数校验采用新成功标准的保存点。

本次发布已完成独立解压、所有打包文件校验和外部输入的完整导入校验；打包和上传没有启动额外 GPU 任务。
