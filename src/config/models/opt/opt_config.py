from src.config.util.base_config import _Arg, _BaseConfig

class ModelConfig(_BaseConfig):
    # IMPORTANT: this is the key used as conf.model_configs.<config_name>
    config_name = "opt"

    def __init__(self) -> None:
        super().__init__()

        # This must match what your OPT model code expects:
        # getattr(conf.model_configs.opt, "hf_model_name", "facebook/opt-350m")
        self._arg_hf_model_name = _Arg(
            type=str,
            help="HuggingFace OPT model name or path.",
            default="facebook/opt-350m",
        )