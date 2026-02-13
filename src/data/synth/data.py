# src/data/synth/data.py

import torch.utils.data as data
import src.config as config

from .synth import SyntheticData, gen_AutoModelForCausalLM

data_load_name = "synth"


class _Info:
    """
    Minimal 'info' object to satisfy milabench synth generators.
    They expect:
      - info.config.vocab_size
      - info.train_length
    """
    def __init__(self, vocab_size: int, train_length: int):
        class _Cfg:
            pass

        self.config = _Cfg()
        self.config.vocab_size = int(vocab_size)
        self.train_length = int(train_length)


def load_data(conf: config.Config) -> data.Dataset:
    """
    Loads a synthetic dataset that yields dict samples like:
      {"input_ids": Tensor[L], "labels": Tensor[L]}
    Controlled by CLI flags under: --data_configs.synth.*
    """

    # Prefer config namespace: conf.data_configs.synth.*
    # Fallback to direct conf.* (in case your config system differs)
    synth_cfg = None
    if hasattr(conf, "data_configs") and hasattr(conf.data_configs, "synth"):
        synth_cfg = conf.data_configs.synth

    def _get(name: str, default):
        if synth_cfg is not None and hasattr(synth_cfg, name):
            return getattr(synth_cfg, name)
        return getattr(conf, name, default)

    vocab_size = int(_get("vocab_size", 50272))  # OPT vocab size (common)
    seq_len = int(_get("seq_len", 512))
    n = int(_get("n", 1024))
    repeat = int(_get("repeat", 1))

    info = _Info(vocab_size=vocab_size, train_length=seq_len)

    # Generator for causal LM: returns input_ids and labels
    gens = gen_AutoModelForCausalLM(info)

    return SyntheticData(generators=gens, n=n, repeat=repeat)
