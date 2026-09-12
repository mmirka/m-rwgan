#!/usr/bin/env python3
"""Train one of the three frozen reward networks M-RWGAN is steered by.

The three targets: saturation (GCN), power (GCN), area (CNN).

See ``src/mrwgan/reward_training.py`` for the shared data-loading/normalization
logic and the architecture differences (GCN for saturation/power, CNN for area).

Requires a 10k-sample dataset (a pickled list of
``mrwgan.data_types.Data_AX``); see ``data/raw/README.md`` for the datasets
used in the paper. Trained checkpoints matching this script's expected output
are already included at ``data/reference/reward_networks/checkpoints/``
(hotspot30 and uniform traffic), so training from scratch is only needed if
you want to retrain on new data.

Example
-------
    python scripts/train_reward_module.py --target saturation \\
        --dataset data/my_dataset_10k_hotspot30_3c.pkl
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mrwgan.reward_training import (  # noqa: E402
    TARGETS,
    RewardTrainConfig,
    train_cnn_reward_net,
    train_gcn_reward_net,
)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--target",
        required=True,
        choices=TARGETS,
        help="Which reward to train: saturation/power (GCN) or area (CNN).",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to a pickled list of mrwgan.data_types.Data_AX. Not included in this repo.",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--dropout-rate", type=float, default=0.2)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--mesh-rows", type=int, default=8)
    parser.add_argument("--mesh-cols", type=int, default=8)
    parser.add_argument("--nb-classes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Checkpoint goes to <output-dir>/checkpoints/reward_<target>/, "
        "training-curve log to <output-dir>/logs/reward_<target>/.",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    config = RewardTrainConfig(
        dataset_path=args.dataset,
        target=args.target,
        output_dir=args.output_dir,
        mesh_rows=args.mesh_rows,
        mesh_cols=args.mesh_cols,
        nb_classes=args.nb_classes,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        dropout_rate=args.dropout_rate,
        train_fraction=args.train_fraction,
        seed=args.seed,
    )

    if not Path(config.dataset_path).exists():
        raise FileNotFoundError(
            f"Training dataset not found: {config.dataset_path}\n"
            "Provide a pickled list of mrwgan.data_types.Data_AX objects for "
            "your 8x8 mesh (see data/raw/README.md for the datasets used in "
            "the paper). Trained checkpoints are already available at "
            "data/reference/reward_networks/checkpoints/."
        )

    train_fn = train_cnn_reward_net if args.target == "area" else train_gcn_reward_net
    result = train_fn(config)

    print("Saved reward network:")
    for key, value in result.items():
        print(f"  {key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
