# BatchSize*_3 Analysis

Source folders:
- `final_experiment_results/BatchSize4_3`
- `final_experiment_results/BatchSize8_3`
- `final_experiment_results/BatchSize16_3`

Fine-grained source folders used:
- Batch 4: `fine_grain_n64_seq1024_2`
- Batch 8: `fine_grain_n64_seq1024_2`
- Batch 16: `fine_grain_n64_seq1024_2`

## Highlights

- Best baseline sample throughput is batch size 16 at 10.63 samples/s.
- Highest average fine-grained GPU utilization is batch size 16 at 99.34%.

## Batch size 4

- Baseline: 300.40 s, 2.413 steps/s, 9.65 samples/s.
- E2E energy: 304.09 s, 0.04914 kWh, overhead 1.23%.
- Fine-grained: 303.70 s, mean step 418.48 ms, forward 132.81 ms, backward 227.65 ms, optimizer 56.72 ms.
- Fine-grained resources: GPU util 99.02%, system CPU util 0.88%, process CPU util 100.80%, GPU mem 11715.94 MB.

## Batch size 8

- Baseline: 300.85 s, 1.296 steps/s, 10.37 samples/s.
- E2E energy: 304.84 s, 0.04953 kWh, overhead 1.33%.
- Fine-grained: 306.10 s, mean step 784.42 ms, forward 265.20 ms, backward 460.79 ms, optimizer 56.81 ms.
- Fine-grained resources: GPU util 99.24%, system CPU util 0.89%, process CPU util 100.78%, GPU mem 13058.77 MB.

## Batch size 16

- Baseline: 301.00 s, 0.664 steps/s, 10.63 samples/s.
- E2E energy: 303.71 s, 0.04955 kWh, overhead 0.90%.
- Fine-grained: 305.10 s, mean step 1524.94 ms, forward 534.15 ms, backward 931.75 ms, optimizer 56.88 ms.
- Fine-grained resources: GPU util 99.34%, system CPU util 0.88%, process CPU util 100.71%, GPU mem 13050.97 MB.

## Notes

- All three batch sizes now use the rerun fine-grained folders so the phase timings reflect logical-batch totals.

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

## Energy View

- `energy_and_emissions.png` now reports GPU-only energy from the CodeCarbon `gpu_energy` field for the E2E energy runs across batch sizes.
