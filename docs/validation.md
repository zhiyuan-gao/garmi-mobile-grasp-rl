# 导出验证与边界

代码整理日期：2026-10-18。独立目录的修改没有覆盖正在运行的原实验代码，也没有启动另一份 GPU 训练。

完成的检查：

- 9 项动作/数据测试：关节增量的单位、裁剪和编码往返；专家阶段保持；扰动坐标系；扰动动作与固定阶段不进入 BC；拒绝混用动作标签版本。
- 7 项真实 Giskard CPU 测试：规划/物理观测坐标一致、倾斜物体目标更新、有限且有界的 QP 动作、复用求解器没有遗留目标、碰撞规则、8 cm 预抓取方向以及扰动隔离。
- 4 项导出集成检查：全部导入文件的哈希、40 个参考数值不变、4000 个评估加 400 个审计场景可查表且互不重复、过滤表仍拒绝越界和错误朝向。
- 新网络入口与 bootstrap 入口的配置生成及 `--check-only` 均通过。
- Python 语法、shell 语法检查通过。USD 离线依赖检查中机器人几何/纹理、碗和厨房文件齐全；机器人材质 `OmniPBR.mdl` 由 Isaac Sim 运行时提供，不在本地资产包中。

没有在整理后的目录重新跑全量 GPU 采集、BC 或 PPO；当前工作站的原始实验仍在运行。新的完整启动入口会先运行物理预检和 BC/PPO 短测，再继续长训。新网络从头学习只是额外入口，不能据此声称已经得到与原初始化相同的成功率。

## 复查命令

导入资产并完成 runtime setup 后：

```bash
bash scripts/rl.sh -m pytest -q \
  /workspace/tests/test_augmentation.py \
  /workspace/tests/test_giskard_data.py \
  /workspace/tests/test_joint_delta.py \
  /workspace/tests/test_export.py

docker exec -e PYTHONPATH=/workspace/rl -e OMP_NUM_THREADS=1 \
  "${GARMI_RL_CONTAINER:-garmi-mobile-grasp-rl}" \
  /home/jovyan/cram-vrb-lab/cognitive_robot_abstract_machine/.venv/bin/python \
  -m pytest -q /workspace/tests/test_giskard_expert.py
```

测试中的单个物理状态快照只用于坐标和求解器回归，不作为训练示范。

## 保留的限制

- 40 个参考都用于训练；没有测试新的物体、碗朝向或未见抓法。
- 状态输入依赖仿真真值，暂时没有感知误差或点云处理。
- 位置过滤只筛家具遮挡的目标姿态，不能证明完整动作可行。
- 原始几何导出资产需要单独提供；本仓库没有完整资产重建流水线。
- 专家纠偏采用动作 MSE 和 PPO，区别于 GA-DDPG 原算法。
- 外层队列可保留完成阶段及成功示范；若在一个 PPO 阶段中断，外层恢复会归档该未完成阶段并从上一阶段检查点重跑。该行为与 PPO 阶段之间的完整状态恢复不同。
