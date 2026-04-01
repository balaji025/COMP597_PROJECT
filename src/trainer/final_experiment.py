if False:
    import time
    from typing import Any, Dict, Optional, Tuple

    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.utils.data as data
    import tqdm.auto

    import src.config as config
    import src.trainer.base as base
    import src.trainer.stats as stats


    class FinalExperimentTrainer(base.Trainer):
        """
        Previous implementation preserved below as requested.
        """

        def __init__(
            self,
            loader: data.DataLoader,
            model: nn.Module,
            optimizer: optim.Optimizer,
            lr_scheduler: Optional[optim.lr_scheduler.LRScheduler],
            device: torch.device,
            stats: stats.TrainerStats,
            conf: Optional[config.Config] = None,
            max_duration_sec: float = 300,
            max_steps: Optional[int] = None,
            enable_checkpointing: bool = False,
            checkpoint_frequency: int = 1,
        ):
            super().__init__(
                model=model,
                loader=loader,
                device=device,
                stats=stats,
                enable_checkpointing=enable_checkpointing,
                checkpoint_frequency=checkpoint_frequency,
            )
            self.optimizer = optimizer
            self.lr_scheduler = lr_scheduler
            self.conf = conf
            self.max_duration_sec = float(max_duration_sec)
            self.max_steps = max_steps

        def checkpoint_dict(self, i: int) -> Dict[str, Any]:
            super_dict = super().checkpoint_dict(i)
            super_dict["optimizer_state_dict"] = self.optimizer.state_dict()
            if self.lr_scheduler is not None:
                super_dict["lr_scheduler_state_dict"] = self.lr_scheduler.state_dict()
            return super_dict

        def _move_batch_to_device(self, batch: Any) -> Any:
            if isinstance(batch, dict):
                return {
                    k: v.to(self.device, non_blocking=True) if torch.is_tensor(v) else v
                    for k, v in batch.items()
                }
            if isinstance(batch, (list, tuple)):
                return type(batch)(
                    x.to(self.device, non_blocking=True) if torch.is_tensor(x) else x
                    for x in batch
                )
            return batch.to(self.device, non_blocking=True) if torch.is_tensor(batch) else batch

        def process_batch(self, i: int, batch: Any) -> Any:
            return self._move_batch_to_device(batch)

        def forward(self, i: int, batch: Any, model_kwargs: Dict[str, Any]) -> torch.Tensor:
            outputs = self.model(**batch, **model_kwargs)
            if hasattr(outputs, "loss"):
                return outputs.loss
            if isinstance(outputs, dict) and "loss" in outputs:
                return outputs["loss"]
            raise ValueError("Model output does not contain a `.loss` field.")

        def backward(self, i: int, loss: torch.Tensor) -> None:
            loss.backward()

        def optimizer_step(self, i: int) -> None:
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        def step(
            self,
            i: int,
            batch: Any,
            model_kwargs: Optional[Dict[str, Any]],
        ) -> Tuple[torch.Tensor, Optional[str]]:
            if model_kwargs is None:
                model_kwargs = {}

            batch = self.process_batch(i, batch)
            self.optimizer.zero_grad(set_to_none=True)

            if hasattr(self.stats, "set_current_step"):
                self.stats.set_current_step(i)

            self.stats.start_forward()
            loss = self.forward(i, batch, model_kwargs)
            self.stats.stop_forward()

            self.stats.start_backward()
            self.backward(i, loss)
            self.stats.stop_backward()

            self.stats.start_optimizer_step()
            self.optimizer_step(i)
            self.stats.stop_optimizer_step()

            return loss, None

        def train(self, model_kwargs: Optional[Dict[str, Any]]) -> None:
            if model_kwargs is None:
                model_kwargs = {}

            start_time = time.perf_counter()
            step_idx = 0
            progress_bar = tqdm.auto.tqdm(desc="loss: N/A", unit="step")

            self.stats.start_train()

            while True:
                elapsed = time.perf_counter() - start_time
                if elapsed >= self.max_duration_sec:
                    break
                if self.max_steps is not None and step_idx >= self.max_steps:
                    break

                for batch in self.loader:
                    elapsed = time.perf_counter() - start_time
                    if elapsed >= self.max_duration_sec:
                        break
                    if self.max_steps is not None and step_idx >= self.max_steps:
                        break

                    self.stats.start_step()
                    loss, descr = self.step(step_idx, batch, model_kwargs)
                    self.stats.stop_step()

                    if self.enable_checkpointing and self.should_save_checkpoint(step_idx):
                        self.stats.start_save_checkpoint()
                        self.save_checkpoint(step_idx)
                        self.stats.stop_save_checkpoint()

                    self.stats.log_loss(loss)
                    self.stats.log_step()

                    progress_bar.set_description(f"loss: {loss.detach().float().item():.4f}")
                    if descr is not None:
                        progress_bar.write(descr)
                    progress_bar.update(1)

                    step_idx += 1

            self.stats.stop_train()
            progress_bar.close()
            self.stats.log_stats()

import os
import signal
import time
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import tqdm.auto

import src.config as config
import src.trainer.base as base
import src.trainer.stats as stats


class FinalExperimentTrainer(base.Trainer):
    """
    Final experiment trainer aligned with the instructor feedback.

    - `baseline_time`: whole-run wall-clock timing only.
    - `e2e_energy`: whole-run wall-clock timing plus whole-run CodeCarbon.
    - `fine_grained`: per-step and per-phase timing plus resource timelines.

    When `max_steps` is provided, it takes priority over the 5-minute cap. This
    supports the intended workflow of calibrating a step count that lasts about
    5 minutes, then re-running all experiment modes with the exact same steps.
    """

    def __init__(
        self,
        loader: data.DataLoader,
        model: nn.Module,
        optimizer: optim.Optimizer,
        lr_scheduler: Optional[optim.lr_scheduler.LRScheduler],
        device: torch.device,
        stats: stats.TrainerStats,
        conf: Optional[config.Config] = None,
        max_duration_sec: float = 300,
        max_steps: Optional[int] = None,
        enable_checkpointing: bool = False,
        checkpoint_frequency: int = 1,
    ):
        super().__init__(
            model=model,
            loader=loader,
            device=device,
            stats=stats,
            enable_checkpointing=enable_checkpointing,
            checkpoint_frequency=checkpoint_frequency,
        )
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.conf = conf
        self.max_duration_sec = float(max_duration_sec)
        self.max_steps = max_steps
        self.experiment_mode = getattr(stats, "experiment_mode", "fine_grained")
        self.collect_fine_grained_metrics = self.experiment_mode == "fine_grained"
        self._termination_requested = False
        self.grad_accum_microbatch_size = 4

    def checkpoint_dict(self, i: int) -> Dict[str, Any]:
        super_dict = super().checkpoint_dict(i)
        super_dict["optimizer_state_dict"] = self.optimizer.state_dict()
        if self.lr_scheduler is not None:
            super_dict["lr_scheduler_state_dict"] = self.lr_scheduler.state_dict()
        return super_dict

    def _move_batch_to_device(self, batch: Any) -> Any:
        if isinstance(batch, dict):
            return {
                k: v.to(self.device, non_blocking=True) if torch.is_tensor(v) else v
                for k, v in batch.items()
            }
        if isinstance(batch, (list, tuple)):
            return type(batch)(
                x.to(self.device, non_blocking=True) if torch.is_tensor(x) else x
                for x in batch
            )
        return batch.to(self.device, non_blocking=True) if torch.is_tensor(batch) else batch

    def process_batch(self, i: int, batch: Any) -> Any:
        return self._move_batch_to_device(batch)

    def _batch_size_of(self, batch: Any) -> int:
        if isinstance(batch, dict):
            for value in batch.values():
                if torch.is_tensor(value) and value.ndim > 0:
                    return int(value.shape[0])
        raise ValueError("Unable to infer batch size for gradient accumulation.")

    def _split_batch(self, batch: Any, microbatch_size: int) -> list[Any]:
        if not isinstance(batch, dict):
            return [batch]

        batch_size = self._batch_size_of(batch)
        if batch_size <= microbatch_size:
            return [batch]

        microbatches: list[Any] = []
        for start in range(0, batch_size, microbatch_size):
            end = min(start + microbatch_size, batch_size)
            microbatches.append(
                {
                    k: (v[start:end] if torch.is_tensor(v) and v.ndim > 0 else v)
                    for k, v in batch.items()
                }
            )
        return microbatches

    def forward(self, i: int, batch: Any, model_kwargs: Dict[str, Any]) -> torch.Tensor:
        outputs = self.model(**batch, **model_kwargs)
        if hasattr(outputs, "loss"):
            return outputs.loss
        if isinstance(outputs, dict) and "loss" in outputs:
            return outputs["loss"]
        raise ValueError("Model output does not contain a `.loss` field.")

    def backward(self, i: int, loss: torch.Tensor) -> None:
        loss.backward()

    def optimizer_step(self, i: int) -> None:
        self.optimizer.step()
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

    def _run_step_without_phase_metrics(
        self,
        i: int,
        batch: Any,
        model_kwargs: Dict[str, Any],
    ) -> torch.Tensor:
        # Previous implementation:
        # loss = self.forward(i, batch, model_kwargs)
        # self.backward(i, loss)
        # self.optimizer_step(i)
        # return loss
        microbatches = self._split_batch(batch, self.grad_accum_microbatch_size)
        accum_scale = 1.0 / len(microbatches)
        last_loss: Optional[torch.Tensor] = None
        for microbatch in microbatches:
            device_batch = self.process_batch(i, microbatch)
            loss = self.forward(i, device_batch, model_kwargs)
            last_loss = loss.detach()
            self.backward(i, loss * accum_scale)
        self.optimizer_step(i)
        if last_loss is None:
            raise RuntimeError("No microbatches were produced for the step.")
        return last_loss

    def _run_step_with_phase_metrics(
        self,
        i: int,
        batch: Any,
        model_kwargs: Dict[str, Any],
    ) -> torch.Tensor:
        # Previous implementation:
        # self.stats.start_forward()
        # loss = self.forward(i, batch, model_kwargs)
        # self.stats.stop_forward()
        #
        # self.stats.start_backward()
        # self.backward(i, loss)
        # self.stats.stop_backward()
        #
        # self.stats.start_optimizer_step()
        # self.optimizer_step(i)
        # self.stats.stop_optimizer_step()
        #
        # return loss
        microbatches = self._split_batch(batch, self.grad_accum_microbatch_size)
        accum_scale = 1.0 / len(microbatches)
        last_loss: Optional[torch.Tensor] = None

        for microbatch in microbatches:
            device_batch = self.process_batch(i, microbatch)
            self.stats.start_forward()
            loss = self.forward(i, device_batch, model_kwargs)
            self.stats.stop_forward()

            self.stats.start_backward()
            self.backward(i, loss * accum_scale)
            self.stats.stop_backward()
            last_loss = loss.detach()

        self.stats.start_optimizer_step()
        self.optimizer_step(i)
        self.stats.stop_optimizer_step()

        if last_loss is None:
            raise RuntimeError("No microbatches were produced for the step.")
        return last_loss

    def step(
        self,
        i: int,
        batch: Any,
        model_kwargs: Optional[Dict[str, Any]],
    ) -> Tuple[torch.Tensor, Optional[str]]:
        if model_kwargs is None:
            model_kwargs = {}

        self.optimizer.zero_grad(set_to_none=True)

        if hasattr(self.stats, "set_current_step"):
            self.stats.set_current_step(i)

        if self.collect_fine_grained_metrics:
            loss = self._run_step_with_phase_metrics(i, batch, model_kwargs)
        else:
            loss = self._run_step_without_phase_metrics(i, batch, model_kwargs)

        return loss, None

    def _should_stop_for_time(self, start_time: float) -> bool:
        if self.max_steps is not None:
            return False
        return (time.perf_counter() - start_time) >= self.max_duration_sec

    def _should_stop(self, start_time: float, step_idx: int) -> bool:
        if self._termination_requested:
            return True
        if self._should_stop_for_time(start_time):
            return True
        return self.max_steps is not None and step_idx >= self.max_steps

    def _install_sigterm_handler(self):
        previous_handler = signal.getsignal(signal.SIGTERM)

        def _handle_sigterm(signum, frame):
            self._termination_requested = True

        signal.signal(signal.SIGTERM, _handle_sigterm)
        return previous_handler

    def _write_debug_timestamp(self, name: str, extra: Optional[Dict[str, Any]] = None) -> None:
        output_dir = getattr(self.stats, "output_dir", None)
        if output_dir is None:
            return

        try:
            os.makedirs(output_dir, exist_ok=True)
            path = os.path.join(output_dir, name)
            with open(path, "w", encoding="utf-8") as f:
                f.write(f"timestamp_iso={datetime.now().isoformat()}\n")
                f.write(f"timestamp_perf_counter={time.perf_counter()}\n")
                f.write(f"experiment_mode={self.experiment_mode}\n")
                f.write(f"max_steps={self.max_steps}\n")
                if extra is not None:
                    for key, value in extra.items():
                        f.write(f"{key}={value}\n")
        except Exception:
            pass

    def train(self, model_kwargs: Optional[Dict[str, Any]]) -> None:
        if model_kwargs is None:
            model_kwargs = {}

        start_time = time.perf_counter()
        step_idx = 0
        progress_bar = tqdm.auto.tqdm(unit="step")
        previous_sigterm_handler = self._install_sigterm_handler()
        started_stats = False
        stopped_stats = False

        try:
            self.stats.start_train()
            started_stats = True
            self._write_debug_timestamp("run_started.txt")

            while True:
                if self._should_stop(start_time, step_idx):
                    break

                for batch in self.loader:
                    if self._should_stop(start_time, step_idx):
                        break

                    if self.collect_fine_grained_metrics:
                        self.stats.start_step()

                    loss, descr = self.step(step_idx, batch, model_kwargs)

                    if self.collect_fine_grained_metrics:
                        self.stats.stop_step()

                    if self.enable_checkpointing and self.should_save_checkpoint(step_idx):
                        self.stats.start_save_checkpoint()
                        self.save_checkpoint(step_idx)
                        self.stats.stop_save_checkpoint()

                    self.stats.log_loss(loss)
                    self.stats.log_step()

                    if descr is not None:
                        progress_bar.write(descr)
                    progress_bar.update(1)
                    step_idx += 1
        finally:
            try:
                self._write_debug_timestamp(
                    "run_loop_exited.txt",
                    {
                        "steps_completed": step_idx,
                        "termination_requested": self._termination_requested,
                    },
                )
                if started_stats and not stopped_stats:
                    self.stats.stop_train()
                    stopped_stats = True
                    self._write_debug_timestamp(
                        "run_stop_train_completed.txt",
                        {
                            "steps_completed": step_idx,
                            "termination_requested": self._termination_requested,
                        },
                    )
                if started_stats:
                    self.stats.log_stats()
                    self._write_debug_timestamp(
                        "run_log_stats_completed.txt",
                        {
                            "steps_completed": step_idx,
                            "termination_requested": self._termination_requested,
                        },
                    )
            finally:
                progress_bar.close()
                signal.signal(signal.SIGTERM, previous_sigterm_handler)
