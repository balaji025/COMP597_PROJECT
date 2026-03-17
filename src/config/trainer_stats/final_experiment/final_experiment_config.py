from src.config.util.base_config import _Arg, _BaseConfig

config_name = "final_experiment"


class TrainerStatsConfig(_BaseConfig):
    def __init__(self) -> None:
        super().__init__()
        self._arg_output_dir = _Arg(
            type=str,
            help="The path of the output directory where final experiment files will be saved.",
            default="final_experiment_results",
        )
        self._arg_run_name = _Arg(
            type=str,
            help="Prefix used for output files.",
            default="run_0",
        )
        self._arg_experiment_mode = _Arg(
            type=str,
            help="Experiment mode: baseline_time, e2e_energy, or fine_grained.",
            default="fine_grained",
        )
        self._arg_sample_interval_sec = _Arg(
            type=float,
            help="Sampling interval in seconds for fine-grained measurements.",
            default=0.5,
        )