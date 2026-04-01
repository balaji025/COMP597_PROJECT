# Multi-Run Experiment Analysis

This report aggregates the current runs from `BatchSize4`, `BatchSize8`, and `BatchSize16`.

## Data source

- BatchSize4: `baseline`, `baseline2`, `baseline3`, `e2e_energy*`, `fine_grain*`
- BatchSize8: `baseline_time`, `baseline2`, `baseline3`, `e2e_energy*`, `fine_grain*`
- BatchSize16: `baseline_time`, `baseline2`, `baseline3`, `e2e_energy*`, `fine_grain*`

## Key observations

- Baseline sample throughput scales by 1.23x from batch 4 to 8 and 1.06x from batch 8 to 16.
- E2E energy changes sample throughput versus baseline by bs=4: -2.01%, bs=8: -1.57%, bs=16: -0.81%.
- Fine-grained changes sample throughput versus baseline by bs=4: -2.08%, bs=8: -1.56%, bs=16: -1.82%.
- Wall-clock times are close but not identical, so throughput remains the cleanest comparison metric across these fixed-budget runs.
- Whole-run energy is lowest at batch size 4 (0.04833 kWh mean) and highest at batch size 16 (0.04940 kWh mean).
- In fine-grained runs, `backward` is the dominant phase on average at 250.56 ms per step.

## GPU Utilization Commentary

- Batch size 4: average GPU utilization is about 97.77% across fine-grained runs, with run-level averages as low as 97.72%. This indicates short windows where the GPU is not fully saturated, so these are the places to inspect for energy-efficiency opportunities.
- Batch size 4: CPU utilization stays low at about 0.93% on average, which is consistent with the workload being primarily GPU-bound rather than CPU-bound.
- Batch size 8: average GPU utilization is about 98.32% across fine-grained runs, with run-level averages as low as 98.06%. This indicates short windows where the GPU is not fully saturated, so these are the places to inspect for energy-efficiency opportunities.
- Batch size 8: CPU utilization stays low at about 1.24% on average, which is consistent with the workload being primarily GPU-bound rather than CPU-bound.
- Batch size 16: average GPU utilization is about 98.70% across fine-grained runs, with run-level averages as low as 98.59%. This indicates short windows where the GPU is not fully saturated, so these are the places to inspect for energy-efficiency opportunities.
- Batch size 16: CPU utilization stays low at about 1.09% on average, which is consistent with the workload being primarily GPU-bound rather than CPU-bound.

## Batch-size summary

- Batch size 4: baseline reaches 16.81 samples/s, E2E energy changes throughput by -2.01%, and fine-grained changes throughput by -2.08%.
- Batch size 8: baseline reaches 20.64 samples/s, E2E energy changes throughput by -1.57%, and fine-grained changes throughput by -1.56%.
- Batch size 16: baseline reaches 21.90 samples/s, E2E energy changes throughput by -0.81%, and fine-grained changes throughput by -1.82%.

## Output files

- `per_run_summary.csv`: one row per current experiment run, including source folders.
- `aggregate_summary.csv`: mean/std metrics grouped by batch size and experiment mode.
- `phase_and_timeline_diagnostics.csv`: fine-grained phase timing plus average resource usage.
- `timeline_batchsize_4.png`, `timeline_batchsize_8.png`, `timeline_batchsize_16.png`: full-run GPU/CPU/memory timelines, one per batch size.
- Other `*.png`: plots for runtime, throughput, energy, throughput deltas, phase timing, and run variability.

## Notes

- Overhead is computed against the matched baseline run with the same batch size and run index.
- `samples_per_second` is the fairest cross-batch efficiency metric because `steps_per_second` decreases as batch size increases.
