#!/usr/bin/env python3
"""Train M-RWGAN: the reward-weighted WGAN-GP generator for 8x8-mesh NoC design.

See ``src/mrwgan/training.py`` for the training loop internals.

Requires an 8x8-mesh training dataset (a pickled list of
``mrwgan.data_types.Data_AX``); see ``data/raw/README.md`` for the datasets
used in the paper, or regenerate one via ``reference/hnocs_pipeline/``.
Point ``--dataset`` at it.

Example
-------
    python scripts/train_mrwgan.py \\
        --dataset data/my_dataset_10k_hotspot30_3c.pkl \\
        --reward-checkpoints data/reference/reward_networks/checkpoints/hotspot30_3classes \\
        --reward-weights 0.5 0.1 0.4 \\
        --epochs 300
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mrwgan.gpu import configure_gpu  # noqa: E402
from mrwgan.training import TrainConfig, train  # noqa: E402


def _find_checkpoint(root: Path, pattern: str) -> str:
    matches = sorted(glob.glob(str(root / pattern)))
    if not matches:
        raise FileNotFoundError(
            f"No reward-network checkpoint matching {pattern!r} found under {root}. "
            "Pass --reward-sat-checkpoint/--reward-pow-checkpoint/--reward-area-checkpoint "
            "explicitly instead of --reward-checkpoints."
        )
    return matches[0]


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to a pickled list of mrwgan.data_types.Data_AX for the 8x8 mesh "
        "(e.g. dataset_10k_hotspot30_3c). Not included in this repo; see data/raw/README.md.",
    )

    reward_group = parser.add_argument_group("reward networks")
    reward_group.add_argument(
        "--reward-checkpoints",
        default="data/reference/reward_networks/checkpoints/hotspot30_3classes",
        help="Directory to auto-discover gcn_*sat*/gcn_*pow*/cnn_*area* checkpoints in "
        "(default: the hotspot30 checkpoints included in this repo). Ignored if the "
        "explicit --reward-*-checkpoint flags are set.",
    )
    reward_group.add_argument("--reward-sat-checkpoint", default=None)
    reward_group.add_argument("--reward-pow-checkpoint", default=None)
    reward_group.add_argument("--reward-area-checkpoint", default=None)
    reward_group.add_argument(
        "--reward-weights",
        type=float,
        nargs=3,
        default=(0.5, 0.1, 0.4),
        metavar=("SAT", "POW", "AREA"),
        help="(sat, pow, area) reward-loss weight triple, each scaled by --beta "
        "(default: 0.5 0.1 0.4).",
    )
    reward_group.add_argument(
        "--beta", type=float, default=10.0, help="Overall reward-loss scale (default: 10)."
    )

    train_group = parser.add_argument_group("training")
    train_group.add_argument("--epochs", type=int, default=300)
    train_group.add_argument("--batch-size", type=int, default=16)
    train_group.add_argument("--latent-dim", type=int, default=100)
    train_group.add_argument("--dropout-rate", type=float, default=0.1)
    train_group.add_argument(
        "--critic-lr", type=float, default=5e-5, help="RMSprop learning rate for both critic and generator."
    )
    train_group.add_argument("--mesh-rows", type=int, default=8)
    train_group.add_argument("--mesh-cols", type=int, default=8)
    train_group.add_argument("--nb-classes", type=int, default=3, help="Router classes: Big/Medium/Small.")
    train_group.add_argument("--seed", type=int, default=None)
    train_group.add_argument(
        "--cpu", action="store_true", help="Force CPU even if a GPU is available."
    )

    out_group = parser.add_argument_group("output")
    out_group.add_argument(
        "--output-dir",
        default="results",
        help="Checkpoints go to <output-dir>/checkpoints/<run-name>/, logs to "
        "<output-dir>/logs/<run-name>/ (default: results/).",
    )
    out_group.add_argument(
        "--run-name",
        default="",
        help="Defaults to mrwgan_sat<W>_pow<W>_area<W> from --reward-weights.",
    )

    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    print(configure_gpu(force_cpu=args.cpu))

    checkpoint_root = Path(args.reward_checkpoints)
    reward_sat = args.reward_sat_checkpoint or _find_checkpoint(checkpoint_root, "gcn_*sat*")
    reward_pow = args.reward_pow_checkpoint or _find_checkpoint(checkpoint_root, "gcn_*pow*")
    reward_area = args.reward_area_checkpoint or _find_checkpoint(checkpoint_root, "cnn_*area*")

    config = TrainConfig(
        dataset_path=args.dataset,
        reward_sat_checkpoint=reward_sat,
        reward_pow_checkpoint=reward_pow,
        reward_area_checkpoint=reward_area,
        output_dir=args.output_dir,
        reward_weights=tuple(args.reward_weights),
        epochs=args.epochs,
        batch_size=args.batch_size,
        latent_dim=args.latent_dim,
        mesh_rows=args.mesh_rows,
        mesh_cols=args.mesh_cols,
        nb_classes=args.nb_classes,
        dropout_rate=args.dropout_rate,
        critic_lr=args.critic_lr,
        beta=args.beta,
        seed=args.seed,
        run_name=args.run_name,
    )

    print(f"Reward checkpoints: sat={reward_sat}  pow={reward_pow}  area={reward_area}")
    artifacts = train(config)
    print("Saved artifacts:")
    for key, value in artifacts.items():
        print(f"  {key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
