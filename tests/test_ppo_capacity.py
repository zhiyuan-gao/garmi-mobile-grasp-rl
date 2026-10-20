from graspgen.train_wrist_ppo import predicted_gpu_memory


def row(envs, load, baseline=20000):
    return dict(command=["--num_envs", str(envs)], gpu_baseline_mib=baseline,
                peak_gpu_used_mib=baseline + load)


def test_large_tier_preserves_fixed_overhead_and_largest_measured_slope():
    # A fixed simulator cost should not double with environment count.
    stages = [row(1024, 5355), row(2048, 6576), row(4096, 11362)]
    assert predicted_gpu_memory(stages, 8192, 20087) == 41021


def test_larger_earlier_growth_is_not_ignored():
    stages = [row(1024, 4000), row(2048, 8000), row(4096, 9000)]
    assert predicted_gpu_memory(stages, 8192, 15000) == 40000


def test_no_distinct_sizes_falls_back_to_proportional_estimate():
    stages = [row(4096, 10000), row(4096, 11000)]
    assert predicted_gpu_memory(stages, 8192, 20000) == 42000
