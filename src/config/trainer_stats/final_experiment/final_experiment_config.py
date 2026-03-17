from src.config.util.base_config import _Arg, _BaseConfig


class FinalExperimentConfig(_BaseConfig):
    def __init__(self) -> None:
        super().__init__()
        self._arg_output_dir = _Arg(
            type=str,
            default="final_experiment_results",
            help="Directory where final experiment outputs are written.",
        )
        self._arg_run_name = _Arg(
            type=str,
            default="run_0",
            help="Prefix for output files.",
        )
        self._arg_experiment_mode = _Arg(
            type=str,
            default="fine_grained",
            help="One of: baseline_time, e2e_energy, fine_grained.",
        )
        self._arg_sample_interval_sec = _Arg(
            type=float,
            default=0.5,
            help="Sampling interval in seconds for fine-grained timelines.",
        )