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

import copy
import os
import time
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import transformers
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
        self.last_num_steps_completed: int = 0
        self.last_total_train_sec: float = 0.0

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

    def _run_step_without_phase_metrics(
        self,
        i: int,
        batch: Any,
        model_kwargs: Dict[str, Any],
    ) -> torch.Tensor:
        loss = self.forward(i, batch, model_kwargs)
        self.backward(i, loss)
        self.optimizer_step(i)
        return loss

    def _run_step_with_phase_metrics(
        self,
        i: int,
        batch: Any,
        model_kwargs: Dict[str, Any],
    ) -> torch.Tensor:
        self.stats.start_forward()
        loss = self.forward(i, batch, model_kwargs)
        self.stats.stop_forward()

        self.stats.start_backward()
        self.backward(i, loss)
        self.stats.stop_backward()

        self.stats.start_optimizer_step()
        self.optimizer_step(i)
        self.stats.stop_optimizer_step()

        return loss

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

        if self.collect_fine_grained_metrics:
            loss = self._run_step_with_phase_metrics(i, batch, model_kwargs)
        else:
            loss = self._run_step_without_phase_metrics(i, batch, model_kwargs)

        return loss, None

    def _should_stop_for_time(self, start_time: float) -> bool:
        if self.max_steps is not None:
            return False
        return (time.perf_counter() - start_time) >= self.max_duration_sec

    def _train_single_run(self, model_kwargs: Optional[Dict[str, Any]]) -> None:
        if model_kwargs is None:
            model_kwargs = {}

        start_time = time.perf_counter()
        step_idx = 0
        progress_bar = tqdm.auto.tqdm(unit="step")

        self.stats.start_train()

        while True:
            if self._should_stop_for_time(start_time):
                break
            if self.max_steps is not None and step_idx >= self.max_steps:
                break

            for batch in self.loader:
                if self._should_stop_for_time(start_time):
                    break
                if self.max_steps is not None and step_idx >= self.max_steps:
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

        self.stats.stop_train()
        progress_bar.close()
        self.stats.log_stats()
        self.last_num_steps_completed = step_idx
        self.last_total_train_sec = time.perf_counter() - start_time

    def _resolve_batch_size_label(self) -> str:
        batch_size = getattr(self.loader, "batch_size", None)
        if batch_size is None and self.conf is not None:
            batch_size = getattr(self.conf, "batch_size", None)
        return f"BS{batch_size}" if batch_size is not None else "BS_unknown"

    def _snapshot_training_state(self) -> Dict[str, Any]:
        snapshot: Dict[str, Any] = {
            "model_state_dict": copy.deepcopy(self.model.state_dict()),
            "optimizer_state_dict": copy.deepcopy(self.optimizer.state_dict()),
            "scheduler_state_dict": copy.deepcopy(self.lr_scheduler.state_dict()) if self.lr_scheduler is not None else None,
        }
        return snapshot

    def _restore_training_state(self, snapshot: Dict[str, Any], num_steps: Optional[int]) -> None:
        self.model.load_state_dict(snapshot["model_state_dict"])
        self.optimizer.load_state_dict(snapshot["optimizer_state_dict"])

        if self.lr_scheduler is not None:
            if num_steps is not None:
                self.lr_scheduler = transformers.get_scheduler(
                    "linear",
                    optimizer=self.optimizer,
                    num_warmup_steps=0,
                    num_training_steps=max(int(num_steps), 1),
                )
            elif snapshot["scheduler_state_dict"] is not None:
                self.lr_scheduler.load_state_dict(snapshot["scheduler_state_dict"])

    def _make_stats_for_run(self, mode: str, output_dir: str, run_name: str) -> stats.TrainerStats:
        if self.conf is None:
            raise ValueError("Automatic experiment orchestration requires a config object.")

        stats_conf = self.conf.trainer_stats_configs.final_experiment
        stats_conf.output_dir = output_dir
        stats_conf.run_name = run_name
        stats_conf.experiment_mode = mode

        new_stats = stats.init_from_conf(self.conf, device=self.device)
        if hasattr(new_stats, "device"):
            new_stats.device = self.device
        return new_stats

    def _run_calibration(self, model_kwargs: Optional[Dict[str, Any]], bs_dir: str, initial_state: Dict[str, Any]) -> int:
        calibration_dir = os.path.join(bs_dir, "calibration")
        os.makedirs(calibration_dir, exist_ok=True)

        self._restore_training_state(initial_state, num_steps=None)
        self.stats = self._make_stats_for_run("baseline_time", calibration_dir, "calibration")

        original_max_steps = self.max_steps
        self.max_steps = None
        self._train_single_run(model_kwargs)
        calibrated_steps = self.last_num_steps_completed
        self.max_steps = original_max_steps

        if calibrated_steps <= 0:
            raise RuntimeError("Calibration run produced zero steps.")
        return calibrated_steps

    def _get_num_repeats(self) -> int:
        if self.conf is None:
            return 3
        stats_conf = getattr(self.conf.trainer_stats_configs, "final_experiment", object())
        return int(getattr(stats_conf, "num_repeats", 3))

    def _run_experiment_suite(self, model_kwargs: Optional[Dict[str, Any]]) -> None:
        if self.conf is None:
            self._train_single_run(model_kwargs)
            return

        stats_conf = self.conf.trainer_stats_configs.final_experiment
        original_output_dir = getattr(stats_conf, "output_dir", "final_experiment_results")
        original_run_name = getattr(stats_conf, "run_name", "run_0")
        original_experiment_mode = getattr(stats_conf, "experiment_mode", "fine_grained")
        original_max_steps = self.max_steps
        original_conf_max_steps = getattr(self.conf, "max_steps", None)

        bs_dir = os.path.join(original_output_dir, self._resolve_batch_size_label())
        os.makedirs(bs_dir, exist_ok=True)

        initial_state = self._snapshot_training_state()
        calibrated_steps = original_max_steps
        if calibrated_steps is None:
            calibrated_steps = self._run_calibration(model_kwargs, bs_dir, initial_state)

        repeats = self._get_num_repeats()
        modes = ["baseline_time", "e2e_energy", "fine_grained"]

        self.max_steps = calibrated_steps
        self.conf.max_steps = calibrated_steps

        try:
            for mode in modes:
                mode_dir = os.path.join(bs_dir, mode)
                os.makedirs(mode_dir, exist_ok=True)

                for repeat_idx in range(1, repeats + 1):
                    run_name = f"run_{repeat_idx}"
                    self._restore_training_state(initial_state, num_steps=calibrated_steps)
                    self.stats = self._make_stats_for_run(mode, mode_dir, run_name)
                    self.experiment_mode = mode
                    self.collect_fine_grained_metrics = mode == "fine_grained"
                    self._train_single_run(model_kwargs)
        finally:
            stats_conf.output_dir = original_output_dir
            stats_conf.run_name = original_run_name
            stats_conf.experiment_mode = original_experiment_mode
            self.max_steps = original_max_steps
            self.conf.max_steps = original_conf_max_steps

    def train(self, model_kwargs: Optional[Dict[str, Any]]) -> None:
        self._run_experiment_suite(model_kwargs)
