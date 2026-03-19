import src.config as config

# This name must match your data loader name (i.e., --data synth)
data_name = "synth"

def add_data_configs(parser: config.ArgumentParser):
    """
    Adds CLI flags under: --data_configs.synth.*
    These will be available on: conf.data_configs.synth.*
    """
    g = parser.add_argument_group("Synthetic dataset (synth)")

    g.add_argument(
        "--data_configs.synth.n",
        type=int,
        default=1000000,  #1024
        help="Number of unique synthetic samples generated and cached.",
    )
    g.add_argument(
        "--data_configs.synth.repeat",
        type=int,
        default=1,
        help="Repeat the cached samples this many times (dataset length = n * repeat).",
    )
    g.add_argument(
        "--data_configs.synth.seq_len",
        type=int,
        default=512,
        help="Sequence length (train_length) for synthetic token sequences.",
    )
    g.add_argument(
        "--data_configs.synth.vocab_size",
        type=int,
        default=50272,
        help="Vocab size for synthetic token IDs (OPT tokenizer vocab is ~50272).",
    )
