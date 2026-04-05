from src.config.util.base_config import _Arg, _BaseConfig


config_name = "final_experiment"


class TrainerConfig(_BaseConfig):
    def __init__(self) -> None:
        super().__init__()
        self._arg_enable_checkpointing = _Arg(
            type=int,
            help="Enable optional checkpoint profiling/saving during final experiments (0 or 1).",
            default=0,
        )
        self._arg_checkpoint_frequency = _Arg(
            type=int,
            help="Save/profile a checkpoint every N steps when checkpointing is enabled.",
            default=100,
        )
