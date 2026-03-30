if False:
    import csv
    import json
    import math
    import os
    import threading
    import time
    from dataclasses import dataclass, field
    from typing import Any, Dict, List, Optional

    import psutil
    import torch

    import src.config as config
    import src.trainer.stats.base as base


    trainer_stats_name = "final_experiment"

import csv
import json
import math
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import psutil
import torch

import src.config as config
import src.trainer.stats.base as base


trainer_stats_name = "final_experiment"

_VALID_EXPERIMENT_MODES = {"baseline_time", "e2e_energy", "fine_grained"}


def construct_trainer_stats(conf: config.Config, **kwargs) -> base.TrainerStats:
    device = kwargs.get("device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    stats_conf = getattr(getattr(conf, "trainer_stats_configs", object()), "final_experiment", object())

    output_dir = getattr(stats_conf, "output_dir", None)
    if output_dir is None:
        try:
            output_dir = conf.trainer_stats_configs.final_experiment.output_dir
        except Exception:
            output_dir = "final_experiment_results"

    experiment_mode = getattr(stats_conf, "experiment_mode", "fine_grained")
    sample_interval_sec = float(getattr(stats_conf, "sample_interval_sec", 0.5))
    run_name = getattr(stats_conf, "run_name", "run_0")

    return FinalExperimentStats(
        device=device,
        output_dir=output_dir,
        run_name=run_name,
        experiment_mode=experiment_mode,
        sample_interval_sec=sample_interval_sec,
    )


def _sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _now_ns() -> int:
    return time.perf_counter_ns()


def _ns_to_ms(x: int) -> float:
    return x / 1_000_000.0


def _mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _std(xs: List[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


@dataclass
class TimerBucket:
    values_ms: List[float] = field(default_factory=list)
    _start_ns: Optional[int] = None

    def start(self, device: torch.device) -> None:
        _sync_if_cuda(device)
        self._start_ns = _now_ns()

    def stop(self, device: torch.device) -> float:
        if self._start_ns is None:
            raise RuntimeError("TimerBucket.stop() called before start().")
        _sync_if_cuda(device)
        end_ns = _now_ns()
        delta_ms = _ns_to_ms(end_ns - self._start_ns)
        self.values_ms.append(delta_ms)
        self._start_ns = None
        return delta_ms


class NvmlHelper:
    def __init__(self) -> None:
        self.enabled = False
        self.handle = None

        try:
            import pynvml  # type: ignore

            self.pynvml = pynvml
            self.pynvml.nvmlInit()
            self.handle = self.pynvml.nvmlDeviceGetHandleByIndex(0)
            self.enabled = True
        except Exception:
            self.pynvml = None
            self.enabled = False
            self.handle = None

    def shutdown(self) -> None:
        if self.enabled and self.pynvml is not None:
            try:
                self.pynvml.nvmlShutdown()
            except Exception:
                pass

    def sample(self) -> Dict[str, Optional[float]]:
        if not self.enabled:
            return {
                "gpu_util_percent": None,
                "gpu_mem_used_mb": None,
                "gpu_power_w": None,
                "gpu_energy_mj": None,
            }

        out: Dict[str, Optional[float]] = {
            "gpu_util_percent": None,
            "gpu_mem_used_mb": None,
            "gpu_power_w": None,
            "gpu_energy_mj": None,
        }

        try:
            util = self.pynvml.nvmlDeviceGetUtilizationRates(self.handle)
            mem = self.pynvml.nvmlDeviceGetMemoryInfo(self.handle)
            out["gpu_util_percent"] = float(util.gpu)
            out["gpu_mem_used_mb"] = float(mem.used) / (1024.0 ** 2)
        except Exception:
            pass

        try:
            power_mw = self.pynvml.nvmlDeviceGetPowerUsage(self.handle)
            out["gpu_power_w"] = float(power_mw) / 1000.0
        except Exception:
            pass

        try:
            energy_mj = self.pynvml.nvmlDeviceGetTotalEnergyConsumption(self.handle)
            out["gpu_energy_mj"] = float(energy_mj)
        except Exception:
            pass

        return out


class CodeCarbonWholeRun:
    def __init__(self, output_dir: str, run_name: str):
        self.tracker = None

        try:
            from codecarbon import EmissionsTracker  # type: ignore

            self.tracker = EmissionsTracker(
                output_dir=output_dir,
                output_file=f"{run_name}_codecarbon.csv",
                measure_power_secs=0.5,
                save_to_file=True,
                log_level="error",
            )
        except Exception:
            self.tracker = None

    def start(self) -> None:
        if self.tracker is not None:
            self.tracker.start()

    def stop(self) -> Optional[float]:
        if self.tracker is not None:
            return self.tracker.stop()
        return None


class TimelineSampler:
    def __init__(
        self,
        sample_interval_sec: float,
        out_csv_path: str,
        get_current_step_fn,
    ):
        self.sample_interval_sec = float(sample_interval_sec)
        self.out_csv_path = out_csv_path
        self.get_current_step_fn = get_current_step_fn
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._rows: List[Dict[str, Any]] = []
        self._nvml = NvmlHelper()
        self._t0: Optional[float] = None

        psutil.cpu_percent(interval=None)

    def start(self) -> None:
        self._t0 = time.perf_counter()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()
        self._flush()
        self._nvml.shutdown()

    def _run(self) -> None:
        while not self._stop_event.is_set():
            now = time.perf_counter()
            elapsed = now - self._t0 if self._t0 is not None else 0.0

            row: Dict[str, Any] = {
                "timestamp_perf_counter_sec": now,
                "elapsed_sec": elapsed,
                "step": self.get_current_step_fn(),
                "cpu_util_percent": psutil.cpu_percent(interval=None),
            }
            row.update(self._nvml.sample())
            self._rows.append(row)

            self._stop_event.wait(self.sample_interval_sec)

    def _flush(self) -> None:
        os.makedirs(os.path.dirname(self.out_csv_path), exist_ok=True)
        fieldnames = [
            "timestamp_perf_counter_sec",
            "elapsed_sec",
            "step",
            "cpu_util_percent",
            "gpu_util_percent",
            "gpu_mem_used_mb",
            "gpu_power_w",
            "gpu_energy_mj",
        ]
        with open(self.out_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self._rows)


class FinalExperimentStats(base.TrainerStats):
    def __init__(
        self,
        device: torch.device,
        output_dir: str,
        run_name: str,
        experiment_mode: str = "fine_grained",
        sample_interval_sec: float = 0.5,
    ):
        if experiment_mode not in _VALID_EXPERIMENT_MODES:
            raise ValueError(f"Unsupported experiment_mode: {experiment_mode}")

        self.device = device
        self.output_dir = output_dir
        self.run_name = run_name
        self.experiment_mode = experiment_mode
        self.sample_interval_sec = sample_interval_sec

        os.makedirs(self.output_dir, exist_ok=True)

        self.train_timer = TimerBucket()
        self.step_timer = TimerBucket()
        self.forward_timer = TimerBucket()
        self.backward_timer = TimerBucket()
        self.optimizer_timer = TimerBucket()
        self.checkpoint_timer = TimerBucket()

        self.losses: List[float] = []
        self.phase_rows: List[Dict[str, Any]] = []
        self.step_rows: List[Dict[str, Any]] = []

        self._current_step: int = -1
        self._current_step_forward_ms: Optional[float] = None
        self._current_step_backward_ms: Optional[float] = None
        self._current_step_optimizer_ms: Optional[float] = None

        self._timeline_sampler: Optional[TimelineSampler] = None
        self._codecarbon: Optional[CodeCarbonWholeRun] = None

    @property
    def collect_fine_grained_metrics(self) -> bool:
        return self.experiment_mode == "fine_grained"

    @property
    def collect_whole_run_energy(self) -> bool:
        return self.experiment_mode == "e2e_energy"

    def set_current_step(self, step: int) -> None:
        self._current_step = int(step)

    def _get_current_step(self) -> int:
        return self._current_step

    def start_train(self) -> None:
        self.train_timer.start(self.device)

        if self.collect_fine_grained_metrics:
            self._timeline_sampler = TimelineSampler(
                sample_interval_sec=self.sample_interval_sec,
                out_csv_path=os.path.join(self.output_dir, f"{self.run_name}_timeline.csv"),
                get_current_step_fn=self._get_current_step,
            )
            self._timeline_sampler.start()
        elif self.collect_whole_run_energy:
            self._codecarbon = CodeCarbonWholeRun(self.output_dir, self.run_name)
            self._codecarbon.start()

    def stop_train(self) -> None:
        total_train_ms = self.train_timer.stop(self.device)

        if self._timeline_sampler is not None:
            self._timeline_sampler.stop()

        emissions_kg = None
        if self._codecarbon is not None:
            emissions_kg = self._codecarbon.stop()

        summary = {
            "experiment_mode": self.experiment_mode,
            "total_train_ms": total_train_ms,
            "num_steps_completed": (self._current_step + 1) if self._current_step >= 0 else 0,
            "num_steps": len(self.step_timer.values_ms) if self.collect_fine_grained_metrics else None,
            "mean_step_ms": _mean(self.step_timer.values_ms) if self.collect_fine_grained_metrics else None,
            "std_step_ms": _std(self.step_timer.values_ms) if self.collect_fine_grained_metrics else None,
            "mean_forward_ms": _mean(self.forward_timer.values_ms) if self.collect_fine_grained_metrics else None,
            "std_forward_ms": _std(self.forward_timer.values_ms) if self.collect_fine_grained_metrics else None,
            "mean_backward_ms": _mean(self.backward_timer.values_ms) if self.collect_fine_grained_metrics else None,
            "std_backward_ms": _std(self.backward_timer.values_ms) if self.collect_fine_grained_metrics else None,
            "mean_optimizer_ms": _mean(self.optimizer_timer.values_ms) if self.collect_fine_grained_metrics else None,
            "std_optimizer_ms": _std(self.optimizer_timer.values_ms) if self.collect_fine_grained_metrics else None,
            "final_loss": self.losses[-1] if self.losses else None,
            "mean_loss": _mean(self.losses) if self.losses else None,
            "codecarbon_emissions_kg": emissions_kg,
        }

        with open(
            os.path.join(self.output_dir, f"{self.run_name}_summary.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(summary, f, indent=2)

        if self.collect_fine_grained_metrics:
            self._write_phase_csv()
            self._write_step_csv()

    def start_step(self) -> None:
        if self.collect_fine_grained_metrics:
            self.step_timer.start(self.device)

    def stop_step(self) -> None:
        if not self.collect_fine_grained_metrics:
            return
        step_ms = self.step_timer.stop(self.device)
        self.step_rows.append(
            {
                "step": self._current_step,
                "step_ms": step_ms,
                "forward_ms": self._current_step_forward_ms,
                "backward_ms": self._current_step_backward_ms,
                "optimizer_ms": self._current_step_optimizer_ms,
            }
        )

    def start_forward(self) -> None:
        if self.collect_fine_grained_metrics:
            self.forward_timer.start(self.device)

    def stop_forward(self) -> None:
        if not self.collect_fine_grained_metrics:
            return
        val = self.forward_timer.stop(self.device)
        self._current_step_forward_ms = val
        self.phase_rows.append({"step": self._current_step, "phase": "forward", "time_ms": val})

    def log_loss(self, loss: torch.Tensor) -> None:
        if self.collect_fine_grained_metrics:
            self.losses.append(float(loss.detach().float().cpu().item()))

    def start_backward(self) -> None:
        if self.collect_fine_grained_metrics:
            self.backward_timer.start(self.device)

    def stop_backward(self) -> None:
        if not self.collect_fine_grained_metrics:
            return
        val = self.backward_timer.stop(self.device)
        self._current_step_backward_ms = val
        self.phase_rows.append({"step": self._current_step, "phase": "backward", "time_ms": val})

    def start_optimizer_step(self) -> None:
        if self.collect_fine_grained_metrics:
            self.optimizer_timer.start(self.device)

    def stop_optimizer_step(self) -> None:
        if not self.collect_fine_grained_metrics:
            return
        val = self.optimizer_timer.stop(self.device)
        self._current_step_optimizer_ms = val
        self.phase_rows.append({"step": self._current_step, "phase": "optimizer", "time_ms": val})

    def start_save_checkpoint(self) -> None:
        if self.collect_fine_grained_metrics:
            self.checkpoint_timer.start(self.device)

    def stop_save_checkpoint(self) -> None:
        if self.collect_fine_grained_metrics:
            self.checkpoint_timer.stop(self.device)

    def log_step(self) -> None:
        pass

    def log_stats(self) -> None:
        pass

    def _write_phase_csv(self) -> None:
        path = os.path.join(self.output_dir, f"{self.run_name}_phase_times.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["step", "phase", "time_ms"])
            writer.writeheader()
            writer.writerows(self.phase_rows)

    def _write_step_csv(self) -> None:
        path = os.path.join(self.output_dir, f"{self.run_name}_step_times.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["step", "step_ms", "forward_ms", "backward_ms", "optimizer_ms"],
            )
            writer.writeheader()
            writer.writerows(self.step_rows)
