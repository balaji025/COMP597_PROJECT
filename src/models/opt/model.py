# === import necessary modules ===
import src.config as config  # Configurations
import src.trainer as trainer  # Trainer base class
import src.trainer.stats as trainer_stats  # Trainer statistics module

# === import necessary external modules ===
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import transformers
from transformers import BitsAndBytesConfig

"""
This file contains the code to train an OPT model using Simple trainer (src/trainer/simple.py).
It is based on the OPT model from HuggingFace Transformers.
https://huggingface.co/docs/transformers/en/model_doc/opt
"""


# -------------------------
# Tokenizer
# -------------------------
def init_opt_tokenizer(model_id: str) -> transformers.PreTrainedTokenizer:
    """
    Initializes the OPT tokenizer from HuggingFace.
    Args:
        model_id (str): HF model id, e.g. "facebook/opt-13b"
    Returns:
        transformers.PreTrainedTokenizer: The tokenizer.
    """
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, use_fast=True)

    # OPT may not have a pad token set; use EOS as pad
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


# -------------------------
# Dataset processing
# -------------------------
def process_dataset(conf: config.Config, tokenizer: transformers.PreTrainedTokenizer, dataset: data.Dataset) -> data.Dataset:
    """
    Tokenizes and formats the dataset for causal LM training.
    Mirrors GPT2 example.
    """
    def tokenize(examples):
        return tokenizer(
            examples["text"],
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

    # Use the same config pattern as GPT2:
    # conf.model_configs.opt.tokenize_num_process
    dataset = dataset.map(
        tokenize,
        batched=True,
        num_proc=conf.model_configs.opt.tokenize_num_process
    )

    # Remove columns that aren't needed by the model
    # (same as your GPT2 example)
    dataset = dataset.remove_columns(column_names=["text", "url", "timestamp"])
    return dataset


# -------------------------
# Optimizer
# -------------------------
def init_opt_optim(conf: config.Config, model: nn.Module) -> optim.Optimizer:
    """
    Initializes AdamW optimizer (same as GPT2 example).
    Note: For big OPT models, you generally want a small LR.
    """
    return optim.AdamW(model.parameters(), lr=conf.learning_rate)


# -------------------------
# Pre-init: model + collator
# -------------------------
def pre_init_opt(conf: config.Config, dataset: data.Dataset) -> Tuple[transformers.PreTrainedModel, data.Dataset, transformers.PreTrainedTokenizer, transformers.DataCollatorForLanguageModeling]:
    """
    Prepares the OPT model, dataset, tokenizer and data collator for training.
    """
    # Choose OPT checkpoint from config if present; else default
    model_id = getattr(conf.model_configs.opt, "hf_model_name", "facebook/opt-1.3b")

    tokenizer = init_opt_tokenizer(model_id)
    dataset = process_dataset(conf, tokenizer, dataset)

    # Causal LM collator (mlm=False)
    data_collator = transformers.DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # 8-bit quantization (like your snippet)
    bnb_config = BitsAndBytesConfig(load_in_8bit=True)

    # IMPORTANT:
    # - use torch_dtype
    # - use device_map="auto"
    # - do NOT .cuda() or .to(device) later
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        attn_implementation="sdpa",
        quantization_config=bnb_config,
        device_map="auto",
    )

    # Ensure pad token id is set
    model.config.pad_token_id = tokenizer.pad_token_id

    return model, dataset, tokenizer, data_collator


# -------------------------
# Trainer
# -------------------------
def simple_trainer(conf: config.Config, model: transformers.PreTrainedModel, dataset: data.Dataset, tokenizer: transformers.PreTrainedTokenizer, data_collator: transformers.DataCollatorForLanguageModeling) -> Tuple[trainer.Trainer, Optional[Dict]]:
    """
    Simple trainer for OPT model. Uses the SimpleTrainer from src/trainer/simple.py.
    Mirrors GPT2 example BUT without model.cuda() (because 8-bit + device_map="auto").
    """
    loader = data.DataLoader(dataset, batch_size=conf.batch_size, collate_fn=data_collator)

    # Do NOT do: model = model.cuda()
    # With 8-bit + device_map="auto", the model is already placed appropriately.
    optimizer = init_opt_optim(conf, model)

    scheduler = transformers.get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=len(loader),
    )

    # Determine a device for the trainer:
    # For device_map models, the "main" device is usually where the first parameter lives.
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")

    return trainer.SimpleTrainer(
        loader=loader,
        model=model,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        device=device,
        stats=trainer_stats.init_from_conf(conf),
    ), {"model_id": getattr(conf.model_configs.opt, "hf_model_name", "facebook/opt-13b")}


# -------------------------
# Entry point
# -------------------------
def opt_init(conf: config.Config, dataset: data.Dataset) -> Tuple[trainer.Trainer, Optional[Dict]]:
    """
    Initializes the OPT model and returns the appropriate trainer based on the configuration.
    Same structure as gpt2_init.
    """
    model, dataset, tokenizer, data_collator = pre_init_opt(conf, dataset)

    if conf.trainer == "simple":
        return simple_trainer(conf, model, dataset, tokenizer, data_collator)
    else:
        raise Exception(f"Unknown trainer type {conf.trainer}")

