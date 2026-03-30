if False:
    # Previous implementation intentionally kept disabled in this file so your
    # current progress is not lost. The new implementation starts below.
    pass

import src.config as config
import src.trainer as trainer
import src.trainer.stats as trainer_stats

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import transformers
from transformers import BitsAndBytesConfig


def init_opt_tokenizer(model_id: str) -> transformers.PreTrainedTokenizer:
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _is_hf_text_dataset(ds: Any) -> bool:
    return hasattr(ds, "map") and hasattr(ds, "column_names")


def process_dataset(
    conf: config.Config,
    tokenizer: transformers.PreTrainedTokenizer,
    dataset: Any,
) -> Any:
    opt_conf = getattr(getattr(conf, "model_configs", object()), "opt", object())
    max_len = getattr(opt_conf, "max_length", 512)
    num_proc = getattr(opt_conf, "tokenize_num_process", 1)

    def tokenize(examples):
        return tokenizer(
            examples["text"],
            max_length=max_len,
            padding="max_length",
            truncation=True,
        )

    dataset = dataset.map(tokenize, batched=True, num_proc=num_proc)

    cols_to_remove = [c for c in ["text", "url", "timestamp"] if c in dataset.column_names]
    if cols_to_remove:
        dataset = dataset.remove_columns(cols_to_remove)

    keep = {"input_ids", "attention_mask"}
    remove = [c for c in dataset.column_names if c not in keep]
    if remove:
        dataset = dataset.remove_columns(remove)

    return dataset


def init_opt_optim(conf: config.Config, model: nn.Module) -> optim.Optimizer:
    lr = getattr(conf, "learning_rate", None)
    if lr is None and hasattr(conf, "trainer_configs") and hasattr(conf.trainer_configs, "simple"):
        lr = getattr(conf.trainer_configs.simple, "learning_rate", None)
    if lr is None:
        lr = 1e-5
    return optim.AdamW(model.parameters(), lr=lr)


def _synth_collate(batch):
    input_ids = torch.stack([x["input_ids"] for x in batch])
    labels = torch.stack([x.get("labels", x["input_ids"]) for x in batch])
    attention_mask = torch.ones_like(input_ids)
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


def pre_init_opt(
    conf: config.Config,
    dataset: Any,
) -> Tuple[
    transformers.PreTrainedModel,
    Any,
    transformers.PreTrainedTokenizer,
    Optional[transformers.DataCollatorForLanguageModeling],
]:
    model_id = getattr(conf.model_configs.opt, "hf_model_name", "facebook/opt-350m")

    tokenizer = init_opt_tokenizer(model_id)

    hf_collator: Optional[transformers.DataCollatorForLanguageModeling] = None
    if _is_hf_text_dataset(dataset):
        dataset = process_dataset(conf, tokenizer, dataset)
        hf_collator = transformers.DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    use_8bit = getattr(conf.model_configs.opt, "load_in_8bit", False)

    if use_8bit:
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            attn_implementation="sdpa",
            quantization_config=bnb_config,
            device_map="auto",
        )
    else:
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_id,
            attn_implementation="sdpa",
        )
        if torch.cuda.is_available():
            model = model.to("cuda")

    model.config.pad_token_id = tokenizer.pad_token_id
    return model, dataset, tokenizer, hf_collator


def _resolve_batch_size(conf: config.Config) -> int:
    batch_size = getattr(conf, "batch_size", None)
    if batch_size is None and hasattr(conf, "trainer_configs") and hasattr(conf.trainer_configs, "simple"):
        batch_size = getattr(conf.trainer_configs.simple, "batch_size", None)
    if batch_size is None and hasattr(conf, "data_configs") and hasattr(conf.data_configs, "synth"):
        batch_size = getattr(conf.data_configs.synth, "batch_size", None)
    if batch_size is None and hasattr(conf, "data_configs") and hasattr(conf.data_configs, "opt"):
        batch_size = getattr(conf.data_configs.opt, "batch_size", None)
    if batch_size is None:
        batch_size = 4
    return int(batch_size)


def _resolve_num_workers(conf: config.Config) -> int:
    if hasattr(conf, "data_configs") and hasattr(conf.data_configs, "synth"):
        return int(getattr(conf.data_configs.synth, "num_workers", 0))
    if hasattr(conf, "data_configs") and hasattr(conf.data_configs, "opt"):
        return int(getattr(conf.data_configs.opt, "num_workers", 0))
    return 0


def _resolve_scheduler_steps(conf: config.Config, loader: data.DataLoader) -> int:
    max_steps = getattr(conf, "max_steps", None)
    if max_steps is not None:
        return max(int(max_steps), 1)
    return max(len(loader), 1)


def _build_common_training_objects(
    conf: config.Config,
    model: transformers.PreTrainedModel,
    dataset: Any,
    hf_collator: Optional[transformers.DataCollatorForLanguageModeling],
):
    batch_size = _resolve_batch_size(conf)
    num_workers = _resolve_num_workers(conf)

    is_hf = _is_hf_text_dataset(dataset)
    collate_fn = hf_collator if (is_hf and hf_collator is not None) else _synth_collate

    loader = data.DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )

    optimizer = init_opt_optim(conf, model)
    scheduler_steps = _resolve_scheduler_steps(conf, loader)
    scheduler = transformers.get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=scheduler_steps,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not hasattr(model, "hf_device_map"):
        model = model.to(device)

    print("torch.cuda.is_available =", torch.cuda.is_available())
    print("OPT model param device =", next(model.parameters()).device)

    stats = trainer_stats.init_from_conf(conf, device=device)
    if hasattr(stats, "device"):
        stats.device = device
    elif hasattr(stats, "_device"):
        stats._device = device

    return loader, model, optimizer, scheduler, device, stats


def simple_trainer(
    conf: config.Config,
    model: transformers.PreTrainedModel,
    dataset: Any,
    tokenizer: transformers.PreTrainedTokenizer,
    hf_collator: Optional[transformers.DataCollatorForLanguageModeling],
) -> Tuple[trainer.Trainer, Optional[Dict]]:
    loader, model, optimizer, scheduler, device, stats = _build_common_training_objects(
        conf, model, dataset, hf_collator
    )

    return trainer.SimpleTrainer(
        loader=loader,
        model=model,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        device=device,
        stats=stats,
    ), {"model_id": getattr(conf.model_configs.opt, "hf_model_name", "facebook/opt-350m")}


def final_experiment_trainer(
    conf: config.Config,
    model: transformers.PreTrainedModel,
    dataset: Any,
    tokenizer: transformers.PreTrainedTokenizer,
    hf_collator: Optional[transformers.DataCollatorForLanguageModeling],
) -> Tuple[trainer.Trainer, Optional[Dict]]:
    loader, model, optimizer, scheduler, device, stats = _build_common_training_objects(
        conf, model, dataset, hf_collator
    )

    max_duration_sec = getattr(conf, "max_duration_sec", 300.0)
    max_steps = getattr(conf, "max_steps", None)

    return trainer.FinalExperimentTrainer(
        loader=loader,
        model=model,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        device=device,
        stats=stats,
        conf=conf,
        max_duration_sec=max_duration_sec,
        max_steps=max_steps,
    ), {"model_id": getattr(conf.model_configs.opt, "hf_model_name", "facebook/opt-350m")}


def opt_init(conf: config.Config, dataset: Any) -> Tuple[trainer.Trainer, Optional[Dict]]:
    model, dataset, tokenizer, hf_collator = pre_init_opt(conf, dataset)

    if conf.trainer == "simple":
        return simple_trainer(conf, model, dataset, tokenizer, hf_collator)

    if conf.trainer == "final_experiment":
        return final_experiment_trainer(conf, model, dataset, tokenizer, hf_collator)

    raise Exception(f"Unknown trainer type {conf.trainer}")
