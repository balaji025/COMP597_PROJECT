# Final Results Spread-Out Analysis

This folder presents the `final_results` outputs in a more spread-out layout similar to `batchsize_3_analysis`.

## Highlights

- Best baseline sample throughput is batch size 16 at 21.76 samples/s.
- Highest average fine-grained GPU utilization is batch size 16 at 99.42%.
- Lowest E2E energy is batch size 4 at 0.04775 kWh.

## Batch size 4

- Baseline: 300.30 s, 4.174 steps/s, 16.69 samples/s.
- E2E energy: 302.35 s, 0.04775 kWh, overhead 0.68%.
- Fine-grained: 302.73 s, mean step 241.06 ms, forward 65.85 ms, backward 117.40 ms, optimizer 56.62 ms.
- Fine-grained resources: GPU util 98.29%, system CPU util 1.52%, process CPU util 100.95%, GPU mem 8757.25 MB.

## Batch size 8

- Baseline: 300.48 s, 2.584 steps/s, 20.67 samples/s.
- E2E energy: 304.23 s, 0.04908 kWh, overhead 1.25%.
- Fine-grained: 302.57 s, mean step 389.37 ms, forward 123.97 ms, backward 207.53 ms, optimizer 56.72 ms.
- Fine-grained resources: GPU util 98.95%, system CPU util 1.07%, process CPU util 100.89%, GPU mem 11715.94 MB.

## Batch size 16

- Baseline: 300.73 s, 1.360 steps/s, 21.76 samples/s.
- E2E energy: 303.31 s, 0.04947 kWh, overhead 0.86%.
- Fine-grained: 300.51 s, mean step 728.84 ms, forward 248.89 ms, backward 422.18 ms, optimizer 56.55 ms.
- Fine-grained resources: GPU util 99.42%, system CPU util 1.02%, process CPU util 100.80%, GPU mem 19341.33 MB.

## Output files

- `summary.csv`
- `phase_summary.csv`
- `timeline_summary.csv`
- `runtime_and_throughput.png`
- `overhead_vs_baseline.png`
- `phase_breakdown.png`
- `resource_trends.png`
- `process_cpu_trend.png`
- `energy_and_emissions.png`
- `timeline_batchsize_4.png`, `timeline_batchsize_8.png`, `timeline_batchsize_16.png`
- `timeline_batchsize_*_gpu_util.png`, `timeline_batchsize_*_system_cpu.png`, `timeline_batchsize_*_process_cpu.png`, `timeline_batchsize_*_gpu_mem.png`, `timeline_batchsize_*_gpu_power.png`
