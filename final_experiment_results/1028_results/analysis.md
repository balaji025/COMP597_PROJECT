# BatchSize4_2 Analysis

Source folders:
- `final_experiment_results/BatchSize4_2/baseline_time_n64_seq1024`
- `final_experiment_results/BatchSize4_2/e2e_energy_n64_seq1024`
- `final_experiment_results/BatchSize4_2/fine_grain_n64_seq1024`

## Summary

- Baseline timing: 300.41 s, 2.453 steps/s.
- E2E energy: 305.23 s, 2.415 steps/s, 0.04920 kWh, 0.000194 kg CO2eq.
- Fine-grained: 303.32 s, 2.430 steps/s, mean step 283.00 ms.

## Fine-grained resource observations

- Average GPU utilization: 99.33%
- Average process CPU utilization: 100.83%
- Average GPU memory used: 11717.25 MB
- Average GPU power: 223.20 W
- Dominant phase: `backward`

## Output files

- `batchsize4_2_runtime_throughput_overhead.png`
- `batchsize4_2_fine_grain_timeline.png`
- `batchsize4_2_phase_breakdown.png`
