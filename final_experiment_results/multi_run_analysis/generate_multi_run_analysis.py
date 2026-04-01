from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = ROOT / "final_experiment_results"
OUTPUT_DIR = Path(__file__).resolve().parent
BATCH_SIZES = [4, 8, 16]
MODE_ORDER = ["baseline_time", "e2e_energy", "fine_grain"]
MODE_LABELS = {
    "baseline_time": "Baseline timing",
    "e2e_energy": "E2E energy",
    "fine_grain": "Fine-grained",
}
PHASES = ["forward", "backward", "optimizer"]
RUN_DIRS = {
    4: {
        "baseline_time": ["baseline", "baseline2", "baseline3"],
        "e2e_energy": ["e2e_energy", "e2e_energy2", "e2e_energy3"],
        "fine_grain": ["fine_grain", "fine_grain2", "fine_grain3"],
    },
    8: {
        "baseline_time": ["baseline_time", "baseline2", "baseline3"],
        "e2e_energy": ["e2e_energy", "e2e_energy2", "e2e_energy3"],
        "fine_grain": ["fine_grain", "fine_grain2", "fine_grain3"],
    },
    16: {
        "baseline_time": ["baseline_time", "baseline2", "baseline3"],
        "e2e_energy": ["e2e_energy", "e2e_energy2", "e2e_energy3"],
        "fine_grain": ["fine_grain", "fine_grain2", "fine_grain3"],
    },
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def safe_float(value: object) -> float | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    return float(value)


def find_single(path: Path, pattern: str) -> Path | None:
    matches = list(path.glob(pattern))
    if not matches:
        return None
    if len(matches) > 1:
        matches.sort()
    return matches[0]



def load_mean_timeline(batch_size: int) -> pd.DataFrame | None:
    runs: list[pd.DataFrame] = []
    for folder_name in RUN_DIRS[batch_size]["fine_grain"]:
        run_dir = RESULTS_ROOT / f"BatchSize{batch_size}" / folder_name
        timeline_path = find_single(run_dir, "*_timeline.csv")
        if timeline_path is None:
            continue

        timeline = pd.read_csv(timeline_path)
        active = timeline[timeline["step"] >= 0].copy()
        if active.empty:
            continue
        runs.append(active)

    if not runs:
        return None

    max_common_elapsed = min(float(run["elapsed_sec"].max()) for run in runs)
    if max_common_elapsed <= 0:
        return None

    step_sec = 0.5
    grid = np.arange(0.0, max_common_elapsed + (step_sec * 0.5), step_sec)
    frames: list[pd.DataFrame] = []
    metrics = ["gpu_util_percent", "cpu_util_percent", "gpu_mem_used_mb", "gpu_power_w"]

    for run in runs:
        run = run.sort_values("elapsed_sec")
        elapsed = run["elapsed_sec"].to_numpy(dtype=float)
        data = {"elapsed_sec": grid}
        for metric in metrics:
            data[metric] = np.interp(grid, elapsed, run[metric].to_numpy(dtype=float))
        frames.append(pd.DataFrame(data))

    combined = pd.concat(frames, ignore_index=True)
    mean_timeline = combined.groupby("elapsed_sec", as_index=False)[metrics].mean()
    return mean_timeline

def load_runs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    per_run_rows: list[dict] = []
    phase_rows: list[dict] = []
    timeline_rows: list[dict] = []

    for batch_size in BATCH_SIZES:
        batch_root = RESULTS_ROOT / f"BatchSize{batch_size}"
        for mode in MODE_ORDER:
            for run_idx, folder_name in enumerate(RUN_DIRS[batch_size][mode], start=1):
                run_dir = batch_root / folder_name
                summary_path = find_single(run_dir, "*_summary.json")
                if summary_path is None:
                    raise FileNotFoundError(f"Missing summary in {run_dir}")
                summary = load_json(summary_path)

                num_steps = summary.get("num_steps")
                if num_steps is None:
                    num_steps = summary.get("num_steps_completed")
                total_train_ms = float(summary["total_train_ms"])
                num_steps = int(num_steps)

                row = {
                    "batch_size": batch_size,
                    "mode": mode,
                    "mode_label": MODE_LABELS[mode],
                    "run": run_idx,
                    "source_folder": str(run_dir.relative_to(ROOT)),
                    "summary_file": str(summary_path.relative_to(ROOT)),
                    "total_train_ms": total_train_ms,
                    "total_train_sec": total_train_ms / 1000.0,
                    "num_steps": num_steps,
                    "steps_per_second": num_steps / (total_train_ms / 1000.0),
                    "samples_per_second": batch_size * num_steps / (total_train_ms / 1000.0),
                    "mean_step_ms": safe_float(summary.get("mean_step_ms")),
                    "std_step_ms": safe_float(summary.get("std_step_ms")),
                    "mean_forward_ms": safe_float(summary.get("mean_forward_ms")),
                    "std_forward_ms": safe_float(summary.get("std_forward_ms")),
                    "mean_backward_ms": safe_float(summary.get("mean_backward_ms")),
                    "std_backward_ms": safe_float(summary.get("std_backward_ms")),
                    "mean_optimizer_ms": safe_float(summary.get("mean_optimizer_ms")),
                    "std_optimizer_ms": safe_float(summary.get("std_optimizer_ms")),
                    "final_loss": safe_float(summary.get("final_loss")),
                    "mean_loss": safe_float(summary.get("mean_loss")),
                    "energy_consumed_kwh": None,
                    "emissions_kg": safe_float(summary.get("codecarbon_emissions_kg")),
                }

                codecarbon_path = find_single(run_dir, "*_codecarbon.csv")
                if codecarbon_path is not None:
                    codecarbon = pd.read_csv(codecarbon_path)
                    if not codecarbon.empty:
                        row["energy_consumed_kwh"] = safe_float(codecarbon.iloc[0].get("energy_consumed"))
                        if row["emissions_kg"] is None:
                            row["emissions_kg"] = safe_float(codecarbon.iloc[0].get("emissions"))

                per_run_rows.append(row)

                phase_path = find_single(run_dir, "*_phase_times.csv")
                if phase_path is not None:
                    phase_times = pd.read_csv(phase_path)
                    for phase in PHASES:
                        values = phase_times.loc[phase_times["phase"] == phase, "time_ms"]
                        if not values.empty:
                            phase_rows.append(
                                {
                                    "batch_size": batch_size,
                                    "mode": mode,
                                    "run": run_idx,
                                    "phase": phase,
                                    "mean_phase_ms": float(values.mean()),
                                    "std_phase_ms": float(values.std(ddof=1)),
                                }
                            )

                timeline_path = find_single(run_dir, "*_timeline.csv")
                if timeline_path is not None:
                    timeline = pd.read_csv(timeline_path)
                    active = timeline[timeline["step"] >= 0].copy()
                    if not active.empty:
                        timeline_rows.append(
                            {
                                "batch_size": batch_size,
                                "mode": mode,
                                "run": run_idx,
                                "avg_gpu_util_percent": float(active["gpu_util_percent"].mean()),
                                "avg_cpu_util_percent": float(active["cpu_util_percent"].mean()),
                                "avg_gpu_mem_used_mb": float(active["gpu_mem_used_mb"].mean()),
                                "avg_gpu_power_w": float(active["gpu_power_w"].mean()),
                                "peak_gpu_power_w": float(active["gpu_power_w"].max()),
                            }
                        )

    per_run = pd.DataFrame(per_run_rows).sort_values(["batch_size", "mode", "run"]).reset_index(drop=True)
    phase_df = pd.DataFrame(phase_rows).sort_values(["batch_size", "mode", "run", "phase"]).reset_index(drop=True)
    timeline_df = pd.DataFrame(timeline_rows).sort_values(["batch_size", "mode", "run"]).reset_index(drop=True)

    baseline_lookup = (
        per_run[per_run["mode"] == "baseline_time"][["batch_size", "run", "total_train_ms", "steps_per_second", "samples_per_second"]]
        .rename(
            columns={
                "total_train_ms": "baseline_total_train_ms",
                "steps_per_second": "baseline_steps_per_second",
                "samples_per_second": "baseline_samples_per_second",
            }
        )
    )
    per_run = per_run.merge(baseline_lookup, on=["batch_size", "run"], how="left")
    per_run["time_overhead_pct_vs_baseline"] = 100.0 * (
        per_run["total_train_ms"] - per_run["baseline_total_train_ms"]
    ) / per_run["baseline_total_train_ms"]
    per_run["step_throughput_delta_pct_vs_baseline"] = 100.0 * (
        per_run["steps_per_second"] - per_run["baseline_steps_per_second"]
    ) / per_run["baseline_steps_per_second"]
    per_run["sample_throughput_delta_pct_vs_baseline"] = 100.0 * (
        per_run["samples_per_second"] - per_run["baseline_samples_per_second"]
    ) / per_run["baseline_samples_per_second"]

    aggregate = (
        per_run.groupby(["batch_size", "mode", "mode_label"], as_index=False)
        .agg(
            runs=("run", "count"),
            total_train_ms_mean=("total_train_ms", "mean"),
            total_train_ms_std=("total_train_ms", "std"),
            num_steps_mean=("num_steps", "mean"),
            num_steps_std=("num_steps", "std"),
            steps_per_second_mean=("steps_per_second", "mean"),
            steps_per_second_std=("steps_per_second", "std"),
            samples_per_second_mean=("samples_per_second", "mean"),
            samples_per_second_std=("samples_per_second", "std"),
            mean_step_ms_mean=("mean_step_ms", "mean"),
            mean_step_ms_std=("mean_step_ms", "std"),
            final_loss_mean=("final_loss", "mean"),
            final_loss_std=("final_loss", "std"),
            energy_consumed_kwh_mean=("energy_consumed_kwh", "mean"),
            energy_consumed_kwh_std=("energy_consumed_kwh", "std"),
            emissions_kg_mean=("emissions_kg", "mean"),
            emissions_kg_std=("emissions_kg", "std"),
            time_overhead_pct_vs_baseline_mean=("time_overhead_pct_vs_baseline", "mean"),
            time_overhead_pct_vs_baseline_std=("time_overhead_pct_vs_baseline", "std"),
            step_throughput_delta_pct_vs_baseline_mean=("step_throughput_delta_pct_vs_baseline", "mean"),
            step_throughput_delta_pct_vs_baseline_std=("step_throughput_delta_pct_vs_baseline", "std"),
            sample_throughput_delta_pct_vs_baseline_mean=("sample_throughput_delta_pct_vs_baseline", "mean"),
            sample_throughput_delta_pct_vs_baseline_std=("sample_throughput_delta_pct_vs_baseline", "std"),
        )
        .sort_values(["batch_size", "mode"])
        .reset_index(drop=True)
    )

    diagnostics = phase_df.merge(timeline_df, on=["batch_size", "mode", "run"], how="left")
    return per_run, aggregate, diagnostics


def save_tables(per_run: pd.DataFrame, aggregate: pd.DataFrame, diagnostics: pd.DataFrame) -> None:
    per_run.to_csv(OUTPUT_DIR / "per_run_summary.csv", index=False)
    aggregate.to_csv(OUTPUT_DIR / "aggregate_summary.csv", index=False)
    diagnostics.to_csv(OUTPUT_DIR / "phase_and_timeline_diagnostics.csv", index=False)


def plot_overhead(aggregate: pd.DataFrame) -> None:
    pivot_mean = aggregate.pivot(index="batch_size", columns="mode", values="time_overhead_pct_vs_baseline_mean")
    pivot_std = aggregate.pivot(index="batch_size", columns="mode", values="time_overhead_pct_vs_baseline_std")
    x = np.arange(len(BATCH_SIZES))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    for offset, mode in [(-width / 2, "e2e_energy"), (width / 2, "fine_grain")]:
        ax.bar(
            x + offset,
            pivot_mean[mode].reindex(BATCH_SIZES).to_numpy(),
            width=width,
            yerr=pivot_std[mode].reindex(BATCH_SIZES).to_numpy(),
            capsize=4,
            label=MODE_LABELS[mode],
        )
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_xticks(x, [str(b) for b in BATCH_SIZES])
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Time overhead vs baseline (%)")
    ax.set_title("Wall-clock overhead across current runs")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "overhead_vs_baseline.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_throughput(aggregate: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for mode in MODE_ORDER:
        mode_df = aggregate[aggregate["mode"] == mode].sort_values("batch_size")
        axes[0].errorbar(
            mode_df["batch_size"],
            mode_df["steps_per_second_mean"],
            yerr=mode_df["steps_per_second_std"],
            marker="o",
            capsize=4,
            label=MODE_LABELS[mode],
        )
        axes[1].errorbar(
            mode_df["batch_size"],
            mode_df["samples_per_second_mean"],
            yerr=mode_df["samples_per_second_std"],
            marker="o",
            capsize=4,
            label=MODE_LABELS[mode],
        )
    axes[0].set_xlabel("Batch size")
    axes[0].set_ylabel("Steps / second")
    axes[0].set_title("Step throughput")
    axes[1].set_xlabel("Batch size")
    axes[1].set_ylabel("Samples / second")
    axes[1].set_title("Sample throughput")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "throughput_scaling.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_energy(aggregate: pd.DataFrame) -> None:
    energy_df = aggregate[aggregate["mode"] == "e2e_energy"].sort_values("batch_size")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    axes[0].errorbar(
        energy_df["batch_size"],
        energy_df["energy_consumed_kwh_mean"],
        yerr=energy_df["energy_consumed_kwh_std"],
        marker="o",
        capsize=4,
        color="tab:orange",
    )
    axes[0].set_xlabel("Batch size")
    axes[0].set_ylabel("Energy consumed (kWh)")
    axes[0].set_title("Whole-run energy")
    axes[1].errorbar(
        energy_df["batch_size"],
        energy_df["emissions_kg_mean"],
        yerr=energy_df["emissions_kg_std"],
        marker="o",
        capsize=4,
        color="tab:gray",
    )
    axes[1].set_xlabel("Batch size")
    axes[1].set_ylabel("Emissions (kg CO2eq)")
    axes[1].set_title("Whole-run emissions")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "energy_and_emissions.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_throughput_delta(aggregate: pd.DataFrame) -> None:
    delta_df = aggregate[aggregate["mode"] != "baseline_time"].copy()
    x = np.arange(len(BATCH_SIZES))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    for offset, mode in [(-width / 2, "e2e_energy"), (width / 2, "fine_grain")]:
        mode_df = delta_df[delta_df["mode"] == mode].set_index("batch_size").reindex(BATCH_SIZES)
        ax.bar(
            x + offset,
            mode_df["sample_throughput_delta_pct_vs_baseline_mean"].to_numpy(),
            width=width,
            yerr=mode_df["sample_throughput_delta_pct_vs_baseline_std"].to_numpy(),
            capsize=4,
            label=MODE_LABELS[mode],
        )
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_xticks(x, [str(b) for b in BATCH_SIZES])
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Sample throughput delta vs baseline (%)")
    ax.set_title("Instrumentation cost under fixed-time runs")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "throughput_delta_vs_baseline.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_phase_breakdown(diagnostics: pd.DataFrame) -> None:
    fine_df = diagnostics[diagnostics["mode"] == "fine_grain"]
    grouped = (
        fine_df.groupby(["batch_size", "phase"], as_index=False)
        .agg(mean_phase_ms=("mean_phase_ms", "mean"), std_phase_ms=("mean_phase_ms", "std"))
        .sort_values(["batch_size", "phase"])
    )

    fig, axes = plt.subplots(1, len(BATCH_SIZES), figsize=(15, 4.8), sharey=True)
    labels = ["Forward", "Backward", "Optimizer"]
    for ax, batch_size in zip(axes, BATCH_SIZES):
        batch_df = grouped[grouped["batch_size"] == batch_size].set_index("phase").reindex(PHASES)
        ax.bar(labels, batch_df["mean_phase_ms"], yerr=batch_df["std_phase_ms"], capsize=4)
        ax.set_title(f"Batch size {batch_size}")
        ax.set_ylabel("Mean phase time (ms)")
    fig.suptitle("Fine-grained phase timing across current runs")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "phase_breakdown_fine_grain.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_timelines_from_current_runs() -> None:
    for batch_size in BATCH_SIZES:
        mean_timeline = load_mean_timeline(batch_size)
        if mean_timeline is None or mean_timeline.empty:
            continue

        fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
        axes[0].plot(mean_timeline["elapsed_sec"], mean_timeline["gpu_util_percent"], label="GPU util", color="tab:blue")
        axes[0].plot(mean_timeline["elapsed_sec"], mean_timeline["cpu_util_percent"], label="CPU util", color="tab:green")
        axes[0].set_ylabel("Utilization (%)")
        axes[0].set_title(f"Mean resource timeline across 3 runs for batch size {batch_size}")
        axes[0].legend(loc="best")
        axes[1].plot(mean_timeline["elapsed_sec"], mean_timeline["gpu_mem_used_mb"], color="tab:purple")
        axes[1].set_ylabel("GPU memory (MB)")
        axes[2].plot(mean_timeline["elapsed_sec"], mean_timeline["gpu_power_w"], color="tab:red")
        axes[2].set_ylabel("GPU power (W)")
        axes[2].set_xlabel("Elapsed time (s)")
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / f"timeline_batchsize_{batch_size}.png", dpi=220, bbox_inches="tight")
        plt.close(fig)



def plot_cpu_utilization_by_batch() -> None:
    for batch_size in BATCH_SIZES:
        mean_timeline = load_mean_timeline(batch_size)
        if mean_timeline is None or mean_timeline.empty:
            continue

        fig, ax = plt.subplots(figsize=(10.5, 4.8))
        ax.plot(
            mean_timeline["elapsed_sec"],
            mean_timeline["cpu_util_percent"],
            linewidth=1.8,
            color="tab:green",
            label="Mean across 3 runs",
        )
        ax.set_title(f"Mean CPU utilization over time for batch size {batch_size}")
        ax.set_xlabel("Elapsed time (s)")
        ax.set_ylabel("CPU utilization (%)")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / f"cpu_utilization_batchsize_{batch_size}.png", dpi=220, bbox_inches="tight")
        plt.close(fig)


def plot_run_variability(per_run: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), sharey=False)
    for ax, batch_size in zip(axes, BATCH_SIZES):
        subset = per_run[per_run["batch_size"] == batch_size].copy()
        xs: list[str] = []
        ys: list[float] = []
        colors: list[str] = []
        for mode, color in [("baseline_time", "tab:blue"), ("e2e_energy", "tab:orange"), ("fine_grain", "tab:green")]:
            mode_slice = subset[subset["mode"] == mode].sort_values("run")
            xs.extend([f"{MODE_LABELS[mode]}\nR{run}" for run in mode_slice["run"]])
            ys.extend(mode_slice["total_train_sec"].tolist())
            colors.extend([color] * len(mode_slice))
        ax.bar(xs, ys, color=colors)
        ax.set_title(f"Batch size {batch_size}")
        ax.set_ylabel("Run time (s)")
        ax.tick_params(axis="x", labelrotation=45)
    fig.suptitle("Run-to-run variability")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "run_variability.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def build_analysis(aggregate: pd.DataFrame, diagnostics: pd.DataFrame) -> str:
    lines: list[str] = []
    lines.append("# Multi-Run Experiment Analysis")
    lines.append("")
    lines.append("This report aggregates the current runs from `BatchSize4`, `BatchSize8`, and `BatchSize16`.")
    lines.append("")
    lines.append("## Data source")
    lines.append("")
    lines.append("- BatchSize4: `baseline`, `baseline2`, `baseline3`, `e2e_energy*`, `fine_grain*`")
    lines.append("- BatchSize8: `baseline_time`, `baseline2`, `baseline3`, `e2e_energy*`, `fine_grain*`")
    lines.append("- BatchSize16: `baseline_time`, `baseline2`, `baseline3`, `e2e_energy*`, `fine_grain*`")
    lines.append("")
    lines.append("## Key observations")
    lines.append("")

    sample_scaling = (
        aggregate[aggregate["mode"] == "baseline_time"][["batch_size", "samples_per_second_mean"]]
        .sort_values("batch_size")
        .set_index("batch_size")["samples_per_second_mean"]
    )
    scale_4_to_8 = sample_scaling.loc[8] / sample_scaling.loc[4]
    scale_8_to_16 = sample_scaling.loc[16] / sample_scaling.loc[8]
    lines.append(
        f"- Baseline sample throughput scales by {scale_4_to_8:.2f}x from batch 4 to 8 and {scale_8_to_16:.2f}x from batch 8 to 16."
    )

    for mode in ("e2e_energy", "fine_grain"):
        mode_rows = aggregate[aggregate["mode"] == mode].sort_values("batch_size")
        deltas = ", ".join(
            f"bs={int(row.batch_size)}: {row.sample_throughput_delta_pct_vs_baseline_mean:.2f}%"
            for row in mode_rows.itertuples()
        )
        lines.append(f"- {MODE_LABELS[mode]} changes sample throughput versus baseline by {deltas}.")

    lines.append("- Wall-clock times are close but not identical, so throughput remains the cleanest comparison metric across these fixed-budget runs.")

    energy_rows = aggregate[aggregate["mode"] == "e2e_energy"].sort_values("batch_size")
    best_energy = energy_rows.loc[energy_rows["energy_consumed_kwh_mean"].idxmin()]
    worst_energy = energy_rows.loc[energy_rows["energy_consumed_kwh_mean"].idxmax()]
    lines.append(
        f"- Whole-run energy is lowest at batch size {int(best_energy['batch_size'])} ({best_energy['energy_consumed_kwh_mean']:.5f} kWh mean) and highest at batch size {int(worst_energy['batch_size'])} ({worst_energy['energy_consumed_kwh_mean']:.5f} kWh mean)."
    )

    fine_phases = diagnostics[diagnostics["mode"] == "fine_grain"].copy()
    phase_grouped = fine_phases.groupby("phase", as_index=False)["mean_phase_ms"].mean().sort_values("mean_phase_ms", ascending=False)
    top_phase = phase_grouped.iloc[0]
    lines.append(f"- In fine-grained runs, `{top_phase['phase']}` is the dominant phase on average at {top_phase['mean_phase_ms']:.2f} ms per step.")

    lines.append("")
    lines.append("## GPU Utilization Commentary")
    lines.append("")
    for batch_size in BATCH_SIZES:
        batch_diag = diagnostics[(diagnostics["batch_size"] == batch_size) & (diagnostics["mode"] == "fine_grain")]
        if batch_diag.empty:
            continue
        avg_gpu = float(batch_diag["avg_gpu_util_percent"].mean())
        min_avg_gpu = float(batch_diag["avg_gpu_util_percent"].min())
        avg_cpu = float(batch_diag["avg_cpu_util_percent"].mean())
        if avg_gpu < 99.5:
            lines.append(
                f"- Batch size {batch_size}: average GPU utilization is about {avg_gpu:.2f}% across fine-grained runs, with run-level averages as low as {min_avg_gpu:.2f}%. This indicates short windows where the GPU is not fully saturated, so these are the places to inspect for energy-efficiency opportunities."
            )
        else:
            lines.append(
                f"- Batch size {batch_size}: GPU utilization stays very high on average at about {avg_gpu:.2f}%, but it is still not perfectly flat at 100%. Any optimization opportunities are likely brief dips between phases or around host-side coordination."
            )
        lines.append(
            f"- Batch size {batch_size}: CPU utilization stays low at about {avg_cpu:.2f}% on average, which is consistent with the workload being primarily GPU-bound rather than CPU-bound."
        )

    lines.append("")
    lines.append("## Batch-size summary")
    lines.append("")
    for batch_size in BATCH_SIZES:
        batch_rows = aggregate[aggregate["batch_size"] == batch_size].set_index("mode")
        baseline = batch_rows.loc["baseline_time"]
        energy = batch_rows.loc["e2e_energy"]
        fine = batch_rows.loc["fine_grain"]
        lines.append(
            f"- Batch size {batch_size}: baseline reaches {baseline['samples_per_second_mean']:.2f} samples/s, E2E energy changes throughput by {energy['sample_throughput_delta_pct_vs_baseline_mean']:.2f}%, and fine-grained changes throughput by {fine['sample_throughput_delta_pct_vs_baseline_mean']:.2f}%."
        )

    lines.append("")
    lines.append("## Output files")
    lines.append("")
    lines.append("- `per_run_summary.csv`: one row per current experiment run, including source folders.")
    lines.append("- `aggregate_summary.csv`: mean/std metrics grouped by batch size and experiment mode.")
    lines.append("- `phase_and_timeline_diagnostics.csv`: fine-grained phase timing plus average resource usage.")
    lines.append("- `timeline_batchsize_4.png`, `timeline_batchsize_8.png`, `timeline_batchsize_16.png`: full-run GPU/CPU/memory timelines, one per batch size.")
    lines.append("- Other `*.png`: plots for runtime, throughput, energy, throughput deltas, phase timing, and run variability.")
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- Overhead is computed against the matched baseline run with the same batch size and run index.")
    lines.append("- `samples_per_second` is the fairest cross-batch efficiency metric because `steps_per_second` decreases as batch size increases.")
    return "\n".join(lines) + "\n"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.style.use("default")
    per_run, aggregate, diagnostics = load_runs()
    save_tables(per_run, aggregate, diagnostics)
    plot_overhead(aggregate)
    plot_throughput(aggregate)
    plot_energy(aggregate)
    plot_throughput_delta(aggregate)
    plot_phase_breakdown(diagnostics)
    plot_timelines_from_current_runs()
    plot_cpu_utilization_by_batch()
    plot_run_variability(per_run)
    (OUTPUT_DIR / "analysis.md").write_text(build_analysis(aggregate, diagnostics), encoding="utf-8")


if __name__ == "__main__":
    main()
