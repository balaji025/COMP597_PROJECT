# === import necessary modules ===

'''
import src.config as config  # Configurations
import src.trainer as trainer  # Trainer base class
import src.trainer.stats as trainer_stats  # Trainer statistics module

# === import necessary external modules ===
from typing import Dict, Optional, Tuple, Any
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import transformers
from transformers import BitsAndBytesConfig
import torch

"""
This file contains the code to train an OPT model using Simple trainer (src/trainer/simple.py).
It supports TWO dataset types:
1) HuggingFace datasets.Dataset with a "text" column (uses tokenizer + HF collator)
2) Synthetic torch-style dataset that already yields {"input_ids": ..., "labels": ...}
"""

# -------------------------
# Tokenizer
# -------------------------
def init_opt_tokenizer(model_id: str) -> transformers.PreTrainedTokenizer:
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


# -------------------------
# Helpers: detect dataset type
# -------------------------
def _is_hf_text_dataset(ds: Any) -> bool:
    # HF datasets.Dataset typically has .map and .column_names
    return hasattr(ds, "map") and hasattr(ds, "column_names")


# -------------------------
# Dataset processing (HF text dataset only)
# -------------------------
def process_dataset(
    conf: config.Config,
    tokenizer: transformers.PreTrainedTokenizer,
    dataset: Any,
) -> Any:
    """
    Tokenizes and formats an HF dataset that contains a "text" column.
    If it's not an HF dataset, this function should NOT be called.
    """
    max_len = getattr(getattr(conf, "model_configs", object()), "opt", None)
    max_len = getattr(max_len, "max_length", 512)

    def tokenize(examples):
        # Return python lists; collator will create tensors
        return tokenizer(
            examples["text"],
            max_length=max_len,
            padding="max_length",
            truncation=True,
        )

    num_proc = getattr(conf.model_configs.opt, "tokenize_num_process", 1)

    dataset = dataset.map(tokenize, batched=True, num_proc=num_proc)

    # Remove columns robustly (avoid crashing if cols don't exist)
    cols_to_remove = [c for c in ["text", "url", "timestamp"] if c in dataset.column_names]
    if cols_to_remove:
        dataset = dataset.remove_columns(cols_to_remove)

    # Keep only model-relevant columns if present
    keep = {"input_ids", "attention_mask"}
    remove = [c for c in dataset.column_names if c not in keep]
    if remove:
        dataset = dataset.remove_columns(remove)

    return dataset


# -------------------------
# Optimizer
# -------------------------
def init_opt_optim(conf: config.Config, model: nn.Module) -> optim.Optimizer:
    # Prefer conf.learning_rate, fallback to trainer_configs.simple.learning_rate, else 1e-5
    lr = getattr(conf, "learning_rate", None)
    if lr is None and hasattr(conf, "trainer_configs") and hasattr(conf.trainer_configs, "simple"):
        lr = getattr(conf.trainer_configs.simple, "learning_rate", None)
    if lr is None:
        lr = 1e-5
    return optim.AdamW(model.parameters(), lr=lr)


# -------------------------
# Collators
# -------------------------
def _synth_collate(batch):
    """
    For synthetic torch datasets that already return tensors:
    expects dicts with "input_ids" and optionally "labels".
    """
    input_ids = torch.stack([x["input_ids"] for x in batch])
    labels = torch.stack([x.get("labels", x["input_ids"]) for x in batch])
    attention_mask = torch.ones_like(input_ids)
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


# -------------------------
# Pre-init: model + dataset + tokenizer + (optional) hf collator
# -------------------------
def pre_init_opt(
    conf: config.Config,
    dataset: Any,
) -> Tuple[transformers.PreTrainedModel, Any, transformers.PreTrainedTokenizer, Optional[transformers.DataCollatorForLanguageModeling]]:
    """
    Prepares the OPT model, dataset, tokenizer.
    If dataset is HF text dataset -> tokenize + return HF collator
    If dataset is synthetic torch dataset -> skip tokenization; return None collator
    """
    model_id = getattr(conf.model_configs.opt, "hf_model_name", "facebook/opt-350m")

    tokenizer = init_opt_tokenizer(model_id)

    hf_collator: Optional[transformers.DataCollatorForLanguageModeling] = None
    if _is_hf_text_dataset(dataset):
        dataset = process_dataset(conf, tokenizer, dataset)
        # This collator will create labels for causal LM (mlm=False)
        hf_collator = transformers.DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Quantization options (default: off for stability)
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
        # Standard (single-device) load; SimpleTrainer can safely .to(device) if it does
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_id,
            attn_implementation="sdpa",
        )
        if torch.cuda.is_available():
            model = model.to("cuda")

    model.config.pad_token_id = tokenizer.pad_token_id
    return model, dataset, tokenizer, hf_collator


# -------------------------
# Trainer
# -------------------------
def simple_trainer(
    conf: config.Config,
    model: transformers.PreTrainedModel,
    dataset: Any,
    tokenizer: transformers.PreTrainedTokenizer,
    hf_collator: Optional[transformers.DataCollatorForLanguageModeling],
) -> Tuple[trainer.Trainer, Optional[Dict]]:
    """
    Simple trainer for OPT model.

    - If HF text dataset: uses HF collator (creates labels)
    - If synth token dataset: uses _synth_collate
    """
    # Batch size fallback
    batch_size = getattr(conf, "batch_size", None)
    if batch_size is None and hasattr(conf, "trainer_configs") and hasattr(conf.trainer_configs, "simple"):
        batch_size = getattr(conf.trainer_configs.simple, "batch_size", None)
    if batch_size is None:
        batch_size = 2

    is_hf = _is_hf_text_dataset(dataset)
    collate_fn = hf_collator if (is_hf and hf_collator is not None) else _synth_collate

    loader = data.DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)

    optimizer = init_opt_optim(conf, model)

    scheduler = transformers.get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=len(loader),
    )

    # Choose device
    # If model uses device_map="auto", parameters can live on different devices;
    # we give trainer a best-effort "main" device (often first param device).
    # Force cuda when available (SLURM gave you a GPU, so it should be)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Move model to that device (only safe when not using device_map="auto")
    model = model.to(device)

    print("torch.cuda.is_available =", torch.cuda.is_available())
    print("OPT model param device =", next(model.parameters()).device)


    stats = trainer_stats.init_from_conf(conf)

    # Force stats to use the same device as training
    if hasattr(stats, "device"):
        stats.device = device
    elif hasattr(stats, "_device"):
        stats._device = device

    return trainer.SimpleTrainer(
        loader=loader,
        model=model,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        device=device,
        stats=stats,
    ), {"model_id": getattr(conf.model_configs.opt, "hf_model_name", "facebook/opt-350m")}


# -------------------------
# Entry point
# -------------------------
def opt_init(conf: config.Config, dataset: Any) -> Tuple[trainer.Trainer, Optional[Dict]]:
    model, dataset, tokenizer, hf_collator = pre_init_opt(conf, dataset)

    if conf.trainer == "simple":
        return simple_trainer(conf, model, dataset, tokenizer, hf_collator)

    raise Exception(f"Unknown trainer type {conf.trainer}")

'''
# === import necessary modules ===
import src.config as config  # Configurations
import src.trainer as trainer  # Trainer base class
import src.trainer.stats as trainer_stats  # Trainer statistics module

# === import necessary external modules ===
from typing import Dict, Optional, Tuple, Any
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import transformers
from transformers import BitsAndBytesConfig
import torch

"""
This file contains the code to train an OPT model using:
1) Simple trainer
2) FinalExperiment trainer

It supports TWO dataset types:
1) HuggingFace datasets.Dataset with a "text" column (uses tokenizer + HF collator)
2) Synthetic torch-style dataset that already yields {"input_ids": ..., "labels": ...}
"""


# -------------------------
# Tokenizer
# -------------------------
def init_opt_tokenizer(model_id: str) -> transformers.PreTrainedTokenizer:
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


# -------------------------
# Helpers: detect dataset type
# -------------------------
def _is_hf_text_dataset(ds: Any) -> bool:
    # HF datasets.Dataset typically has .map and .column_names
    return hasattr(ds, "map") and hasattr(ds, "column_names")


# -------------------------
# Dataset processing (HF text dataset only)
# -------------------------
def process_dataset(
    conf: config.Config,
    tokenizer: transformers.PreTrainedTokenizer,
    dataset: Any,
) -> Any:
    """
    Tokenizes and formats an HF dataset that contains a "text" column.
    If it's not an HF dataset, this function should NOT be called.
    """
    max_len = getattr(getattr(conf, "model_configs", object()), "opt", None)
    max_len = getattr(max_len, "max_length", 512)

    def tokenize(examples):
        # Return python lists; collator will create tensors
        return tokenizer(
            examples["text"],
            max_length=max_len,
            padding="max_length",
            truncation=True,
        )

    num_proc = getattr(conf.model_configs.opt, "tokenize_num_process", 1)

    dataset = dataset.map(tokenize, batched=True, num_proc=num_proc)

    # Remove columns robustly (avoid crashing if cols don't exist)
    cols_to_remove = [c for c in ["text", "url", "timestamp"] if c in dataset.column_names]
    if cols_to_remove:
        dataset = dataset.remove_columns(cols_to_remove)

    # Keep only model-relevant columns if present
    keep = {"input_ids", "attention_mask"}
    remove = [c for c in dataset.column_names if c not in keep]
    if remove:
        dataset = dataset.remove_columns(remove)

    return dataset


# -------------------------
# Optimizer
# -------------------------
def init_opt_optim(conf: config.Config, model: nn.Module) -> optim.Optimizer:
    # Prefer conf.learning_rate, fallback to trainer_configs.simple.learning_rate, else 1e-5
    lr = getattr(conf, "learning_rate", None)
    if lr is None and hasattr(conf, "trainer_configs") and hasattr(conf.trainer_configs, "simple"):
        lr = getattr(conf.trainer_configs.simple, "learning_rate", None)
    if lr is None:
        lr = 1e-5
    return optim.AdamW(model.parameters(), lr=lr)


# -------------------------
# Collators
# -------------------------
def _synth_collate(batch):
    """
    For synthetic torch datasets that already return tensors:
    expects dicts with "input_ids" and optionally "labels".
    """
    input_ids = torch.stack([x["input_ids"] for x in batch])
    labels = torch.stack([x.get("labels", x["input_ids"]) for x in batch])
    attention_mask = torch.ones_like(input_ids)
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


# -------------------------
# Pre-init: model + dataset + tokenizer + (optional) hf collator
# -------------------------
def pre_init_opt(
    conf: config.Config,
    dataset: Any,
) -> Tuple[
    transformers.PreTrainedModel,
    Any,
    transformers.PreTrainedTokenizer,
    Optional[transformers.DataCollatorForLanguageModeling],
]:
    """
    Prepares the OPT model, dataset, tokenizer.
    If dataset is HF text dataset -> tokenize + return HF collator
    If dataset is synthetic torch dataset -> skip tokenization; return None collator
    """
    model_id = getattr(conf.model_configs.opt, "hf_model_name", "facebook/opt-350m")

    tokenizer = init_opt_tokenizer(model_id)

    hf_collator: Optional[transformers.DataCollatorForLanguageModeling] = None
    if _is_hf_text_dataset(dataset):
        dataset = process_dataset(conf, tokenizer, dataset)
        # This collator will create labels for causal LM (mlm=False)
        hf_collator = transformers.DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Quantization options (default: off for stability)
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
        # Standard (single-device) load
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_id,
            attn_implementation="sdpa",
        )
        if torch.cuda.is_available():
            model = model.to("cuda")

    model.config.pad_token_id = tokenizer.pad_token_id
    return model, dataset, tokenizer, hf_collator


# -------------------------
# Shared loader / optimizer / device / stats
# -------------------------
def _build_common_training_objects(
    conf: config.Config,
    model: transformers.PreTrainedModel,
    dataset: Any,
    hf_collator: Optional[transformers.DataCollatorForLanguageModeling],
):
    # Batch size fallback
    batch_size = getattr(conf, "batch_size", None)
    if batch_size is None and hasattr(conf, "trainer_configs") and hasattr(conf.trainer_configs, "simple"):
        batch_size = getattr(conf.trainer_configs.simple, "batch_size", None)
    if batch_size is None and hasattr(conf, "data_configs") and hasattr(conf.data_configs, "synth"):
        batch_size = getattr(conf.data_configs.synth, "batch_size", None)
    if batch_size is None and hasattr(conf, "data_configs") and hasattr(conf.data_configs, "opt"):
        batch_size = getattr(conf.data_configs.opt, "batch_size", None)
    if batch_size is None:
        batch_size = 2

    print(batch_size)

    num_workers = 0
    if hasattr(conf, "data_configs") and hasattr(conf.data_configs, "synth"):
        num_workers = getattr(conf.data_configs.synth, "num_workers", 0)
    elif hasattr(conf, "data_configs") and hasattr(conf.data_configs, "opt"):
        num_workers = getattr(conf.data_configs.opt, "num_workers", 0)

    is_hf = _is_hf_text_dataset(dataset)
    collate_fn = hf_collator if (is_hf and hf_collator is not None) else _synth_collate

    loader = data.DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )

    print("\n at model.py      \n", batch_size)

    optimizer = init_opt_optim(conf, model)

    scheduler = transformers.get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=max(len(loader), 1),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    print("torch.cuda.is_available =", torch.cuda.is_available())
    print("OPT model param device =", next(model.parameters()).device)

    stats = trainer_stats.init_from_conf(conf)

    # Force stats to use the same device as training
    if hasattr(stats, "device"):
        stats.device = device
    elif hasattr(stats, "_device"):
        stats._device = device

    return loader, model, optimizer, scheduler, device, stats


# -------------------------
# Trainer: Simple
# -------------------------
def simple_trainer(
    conf: config.Config,
    model: transformers.PreTrainedModel,
    dataset: Any,
    tokenizer: transformers.PreTrainedTokenizer,
    hf_collator: Optional[transformers.DataCollatorForLanguageModeling],
) -> Tuple[trainer.Trainer, Optional[Dict]]:
    """
    Simple trainer for OPT model.

    - If HF text dataset: uses HF collator (creates labels)
    - If synth token dataset: uses _synth_collate
    """
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


# -------------------------
# Trainer: Final experiment
# -------------------------
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


# -------------------------
# Entry point
# -------------------------
def opt_init(conf: config.Config, dataset: Any) -> Tuple[trainer.Trainer, Optional[Dict]]:
    model, dataset, tokenizer, hf_collator = pre_init_opt(conf, dataset)

    if conf.trainer == "simple":
        return simple_trainer(conf, model, dataset, tokenizer, hf_collator)

    if conf.trainer == "final_experiment":
        return final_experiment_trainer(conf, model, dataset, tokenizer, hf_collator)

    raise Exception(f"Unknown trainer type {conf.trainer}")