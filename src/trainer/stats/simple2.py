import logging
import src.config as config
import src.trainer.stats.base as base
import src.trainer.stats.utils as utils
import torch
import os
import csv
import datetime

logger = logging.getLogger(__name__)

trainer_stats_name = "simple2"

def construct_trainer_stats(conf: config.Config, **kwargs) -> base.TrainerStats:
    device = kwargs.get("device", None)
    if device is None:
        logger.warning("No device provided to simple trainer stats. Using default PyTorch device")
        device = torch.get_default_device()

    log_dir = kwargs.get("log_dir", "./logs")
    return SimpleTrainerStats(device=device, log_dir=log_dir)


class SimpleTrainerStats(base.TrainerStats):
    def __init__(self, device: torch.device, log_dir: str = "./logs") -> None:
        super().__init__()
        self.device = device

        self.step_stats = utils.RunningTimer()
        self.forward_stats = utils.RunningTimer()
        self.backward_stats = utils.RunningTimer()
        self.optimizer_step_stats = utils.RunningTimer()
        self.save_checkpoint_stats = utils.RunningTimer()

        # Optional psutil (CPU/RAM)
        try:
            import psutil
            self._psutil = psutil
            self._process = psutil.Process(os.getpid())
            # prime cpu_percent so the next call is meaningful
            self._process.cpu_percent(interval=None)
        except Exception:
            self._psutil = None
            self._process = None

        os.makedirs(log_dir, exist_ok=True)

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        pid = os.getpid()
        rank = int(os.environ.get("RANK", "0"))
        self.file_path = os.path.join(log_dir, f"training_metrics_{timestamp}_rank{rank}_pid{pid}.csv")

        self.csv_file = open(self.file_path, mode="w", newline="")
        self.csv_writer = csv.writer(self.csv_file)

        self.csv_writer.writerow([
            "step_ms",
            "forward_ms",
            "backward_ms",
            "optimizer_ms",
            "gpu_mem_alloc_MB",
            "gpu_mem_reserved_MB",
            "gpu_mem_max_MB",
            "proc_cpu_percent",
            "sys_cpu_percent",
            "ram_used_MB",
            "ram_percent",
        ])
        self.csv_file.flush()

    def start_train(self) -> None:
        pass

    def stop_train(self) -> None:
        if not self.csv_file.closed:
            self.csv_file.close()

    def start_step(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.step_stats.start()

    def stop_step(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.step_stats.stop()

        step_ms = self.step_stats.get_last() / 1_000_000
        forward_ms = self.forward_stats.get_last() / 1_000_000
        backward_ms = self.backward_stats.get_last() / 1_000_000
        optimizer_ms = self.optimizer_step_stats.get_last() / 1_000_000

        # GPU memory
        if self.device.type == "cuda":
            idx = self.device.index if self.device.index is not None else torch.cuda.current_device()
            gpu_mem_alloc = torch.cuda.memory_allocated(idx) / (1024**2)
            gpu_mem_reserved = torch.cuda.memory_reserved(idx) / (1024**2)
            gpu_mem_max = torch.cuda.max_memory_allocated(idx) / (1024**2)
        else:
            gpu_mem_alloc = gpu_mem_reserved = gpu_mem_max = 0.0

        # CPU + RAM
        proc_cpu = sys_cpu = ram_used = ram_percent = 0.0
        if self._psutil is not None and self._process is not None:
            try:
                proc_cpu = self._process.cpu_percent(interval=None)
                sys_cpu = self._psutil.cpu_percent(interval=None)
                vm = self._psutil.virtual_memory()
                ram_used = vm.used / (1024**2)
                ram_percent = vm.percent
            except Exception:
                pass

        self.csv_writer.writerow([
            step_ms,
            forward_ms,
            backward_ms,
            optimizer_ms,
            gpu_mem_alloc,
            gpu_mem_reserved,
            gpu_mem_max,
            proc_cpu,
            sys_cpu,
            ram_used,
            ram_percent,
        ])
        self.csv_file.flush()

    def start_forward(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.forward_stats.start()

    def stop_forward(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.forward_stats.stop()

    def start_backward(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.backward_stats.start()

    def stop_backward(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.backward_stats.stop()

    def start_optimizer_step(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.optimizer_step_stats.start()

    def stop_optimizer_step(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.optimizer_step_stats.stop()

    def start_save_checkpoint(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.save_checkpoint_stats.start()

    def stop_save_checkpoint(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.save_checkpoint_stats.stop()

    # keep these in case base class expects them
    def log_step(self) -> None:
        pass

    def log_stats(self) -> None:
        pass

    def log_loss(self, loss: torch.Tensor) -> None:
        pass