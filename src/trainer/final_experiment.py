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

    def train(self, model_kwargs: Optional[Dict[str, Any]]) -> None:
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
