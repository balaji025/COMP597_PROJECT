import src.models.opt.model as opt_model
import src.config as config
import src.trainer as trainer

from typing import Any, Dict, Optional, Tuple
import torch.utils.data as data

model_name = "opt"

def init_model(conf: config.Config, dataset: data.Dataset):
    return opt_model.opt_init(conf, dataset)
