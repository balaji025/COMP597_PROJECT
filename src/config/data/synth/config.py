from src.config.util.base_config import _Arg, _BaseConfig


config_name = "synth"


class DataConfig(_BaseConfig):

    def __init__(self) -> None:
        super().__init__()

        self._arg_n = _Arg(
            type=int,
            default=1000000,
            help="Number of unique synthetic samples generated and cached.",
        )
        self._arg_max_samples = _Arg(
            type=int,
            default=None,
            help="Alias for n. If provided, overrides synth.n.",
        )
        self._arg_repeat = _Arg(
            type=int,
            default=1,
            help="Repeat the cached samples this many times.",
        )
        self._arg_seq_len = _Arg(
            type=int,
            default=512,
            help="Sequence length for synthetic token sequences.",
        )
        self._arg_vocab_size = _Arg(
            type=int,
            default=50272,
            help="Vocabulary size for synthetic token IDs.",
        )
        self._arg_batch_size = _Arg(
            type=int,
            default=None,
            help="Optional synth-specific batch size fallback.",
        )
        self._arg_num_workers = _Arg(
            type=int,
            default=0,
            help="Number of DataLoader workers for synthetic data.",
        )
