from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = ROOT / 'final_experiment_results'
OUTPUT_DIR = Path(__file__).resolve().parent
BATCH_SIZES = [4, 8, 16]


def load_batch_data(batch_size: int) -> dict:
    batch_dir = RESULTS_ROOT / f'BatchSize{batch_size}'
    baseline_dir = batch_dir / 'baseline_time'
    if not baseline_dir.exists():
        baseline_dir = batch_dir / 'baseline'
    energy_dir = batch_dir / 'e2e_energy'
    fine_dir = batch_dir / 'fine_grain'

    baseline_summary = json.loads(next(baseline_dir.glob('*summary.json')).read_text(encoding='utf-8'))
    energy_summary = json.loads(next(energy_dir.glob('*summary.json')).read_text(encoding='utf-8'))
    fine_summary = json.loads(next(fine_dir.glob('*summary.json')).read_text(encoding='utf-8'))
    timeline = pd.read_csv(next(fine_dir.glob('*timeline.csv')))
    phase_times = pd.read_csv(next(fine_dir.glob('*phase_times.csv')))
    codecarbon = pd.read_csv(next(energy_dir.glob('*codecarbon.csv')))

    baseline_ms = baseline_summary['total_train_ms']
    energy_ms = energy_summary['total_train_ms']
    fine_ms = fine_summary['total_train_ms']
    steps = fine_summary['num_steps_completed']

    summary_row = {
        'batch_size': batch_size,
        'baseline_time_ms': baseline_ms,
        'e2e_energy_time_ms': energy_ms,
        'fine_grained_time_ms': fine_ms,
        'steps_completed': steps,
        'steps_per_second': steps / (fine_ms / 1000.0),
        'samples_per_second': batch_size * steps / (fine_ms / 1000.0),
        'e2e_energy_overhead_pct': 100.0 * (energy_ms - baseline_ms) / baseline_ms,
        'fine_grained_overhead_pct': 100.0 * (fine_ms - baseline_ms) / baseline_ms,
        'mean_forward_ms': fine_summary['mean_forward_ms'],
        'std_forward_ms': fine_summary['std_forward_ms'],
        'mean_backward_ms': fine_summary['mean_backward_ms'],
        'std_backward_ms': fine_summary['std_backward_ms'],
        'mean_optimizer_ms': fine_summary['mean_optimizer_ms'],
        'std_optimizer_ms': fine_summary['std_optimizer_ms'],
        'energy_consumed_kwh': float(codecarbon.iloc[0]['energy_consumed']),
        'emissions_kg': energy_summary['codecarbon_emissions_kg'],
    }

    return {
        'summary_row': summary_row,
        'timeline': timeline,
        'phase_times': phase_times,
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    loaded = [load_batch_data(batch_size) for batch_size in BATCH_SIZES]
    df = pd.DataFrame([item['summary_row'] for item in loaded]).sort_values('batch_size').reset_index(drop=True)

    plt.style.use('default')

    x = np.arange(len(df))
    width = 0.35
    plt.figure(figsize=(9, 5))
    plt.bar(x - width / 2, df['e2e_energy_overhead_pct'], width=width, label='E2E energy')
    plt.bar(x + width / 2, df['fine_grained_overhead_pct'], width=width, label='Fine-grained')
    plt.xticks(x, [str(v) for v in df['batch_size']])
    plt.xlabel('Batch size')
    plt.ylabel('Overhead vs baseline (%)')
    plt.title('Measurement overhead by experiment mode')
    plt.axhline(5.0, color='tab:red', linestyle='--', linewidth=1, label='5% target')
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'measurement_overheads.png', dpi=200, bbox_inches='tight')
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(df['batch_size'], df['steps_per_second'], marker='o')
    axes[0].set_xlabel('Batch size')
    axes[0].set_ylabel('Steps / second')
    axes[0].set_title('Step throughput')
    axes[1].plot(df['batch_size'], df['samples_per_second'], marker='o', color='tab:green')
    axes[1].set_xlabel('Batch size')
    axes[1].set_ylabel('Samples / second')
    axes[1].set_title('Sample throughput')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'throughput_scaling.png', dpi=200, bbox_inches='tight')
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].bar(df['batch_size'].astype(str), df['energy_consumed_kwh'], color='tab:orange')
    axes[0].set_xlabel('Batch size')
    axes[0].set_ylabel('Energy consumed (kWh)')
    axes[0].set_title('Whole-run energy from CodeCarbon')
    axes[1].bar(df['batch_size'].astype(str), df['emissions_kg'], color='tab:gray')
    axes[1].set_xlabel('Batch size')
    axes[1].set_ylabel('Emissions (kg CO2eq)')
    axes[1].set_title('Whole-run emissions')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'energy_and_emissions.png', dpi=200, bbox_inches='tight')
    plt.close()

    phases = ['forward', 'backward', 'optimizer']
    labels = ['Forward', 'Backward', 'Optimizer']
    fig, axes = plt.subplots(1, len(loaded), figsize=(15, 4.5), sharey=True)
    for ax, item in zip(axes, loaded):
        batch_size = item['summary_row']['batch_size']
        grouped = item['phase_times'].groupby('phase')['time_ms'].agg(['mean', 'std']).reindex(phases)
        ax.bar(labels, grouped['mean'], yerr=grouped['std'], capsize=4)
        ax.set_title(f'Batch size {batch_size}')
        ax.set_ylabel('Time per phase (ms)')
    fig.suptitle('Average phase time per step')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'phase_times_by_batch.png', dpi=200, bbox_inches='tight')
    plt.close()

    for item in loaded:
        batch_size = item['summary_row']['batch_size']
        timeline = item['timeline']
        active = timeline[timeline['step'] >= 0].copy()
        fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
        axes[0].plot(active['elapsed_sec'], active['gpu_util_percent'], label='GPU util', color='tab:blue')
        axes[0].plot(active['elapsed_sec'], active['cpu_util_percent'], label='CPU util', color='tab:green')
        axes[0].set_ylabel('Utilization (%)')
        axes[0].set_title(f'Resource timeline for batch size {batch_size}')
        axes[0].legend(loc='best')
        axes[1].plot(active['elapsed_sec'], active['gpu_mem_used_mb'], color='tab:purple')
        axes[1].set_ylabel('GPU memory (MB)')
        axes[2].plot(active['elapsed_sec'], active['gpu_power_w'], color='tab:red')
        axes[2].set_ylabel('GPU power (W)')
        axes[2].set_xlabel('Elapsed time (s)')
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f'timeline_batchsize_{batch_size}.png', dpi=200, bbox_inches='tight')
        plt.close()


if __name__ == '__main__':
    main()
