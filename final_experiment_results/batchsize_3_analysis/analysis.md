# BatchSize*_3 Analysis

Source folders:
- `final_experiment_results/BatchSize4_3`
- `final_experiment_results/BatchSize8_3`
- `final_experiment_results/BatchSize16_3`

## Highlights

- Best baseline sample throughput is batch size 16 at 10.63 samples/s.
- Highest average fine-grained GPU utilization is batch size 16 at 99.40%.

## Batch size 4

- Baseline: 300.40 s, 2.413 steps/s, 9.65 samples/s.
- E2E energy: 304.09 s, 0.04914 kWh, overhead 1.23%.
- Fine-grained: 303.77 s, mean step 418.56 ms, final loss 10.7951.
- Fine-grained resources: GPU util 99.04%, process CPU util 0.52%, GPU mem 11715.95 MB.

## Batch size 8

- Baseline: 300.85 s, 1.296 steps/s, 10.37 samples/s.
- E2E energy: 304.84 s, 0.04953 kWh, overhead 1.33%.
- Fine-grained: 302.16 s, mean step 774.28 ms, final loss 10.8300.
- Fine-grained resources: GPU util 99.27%, process CPU util 1.29%, GPU mem 13058.69 MB.

## Batch size 16

- Baseline: 301.00 s, 0.664 steps/s, 10.63 samples/s.
- E2E energy: 303.71 s, 0.04955 kWh, overhead 0.90%.
- Fine-grained: 300.10 s, mean step 1499.93 ms, final loss 10.8901.
- Fine-grained resources: GPU util 99.40%, process CPU util 1.64%, GPU mem 13052.05 MB.

## Output files

- `summary.csv`
- `phase_summary.csv`
- `timeline_summary.csv`
- `runtime_and_throughput.png`
- `overhead_vs_baseline.png`
- `phase_breakdown.png`
- `resource_trends.png`
- `timeline_batchsize_4.png`, `timeline_batchsize_8.png`, `timeline_batchsize_16.png`
