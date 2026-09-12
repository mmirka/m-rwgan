#!/usr/bin/env python3
"""Reproduce Fig 6.10 (GAN generator/critic loss and reward-score curves).

DATE2022 / PhD thesis Chapter 6, Fig 6.10. Two plots:
- Generator vs. critic (discriminator) loss, per training batch:
  `generator_loss[:, 0]` and `discriminator_loss[:, 2]` — both `(n_batches,
  4)` arrays, plotted against a fractional-epoch x-axis
  (`n_epochs * batch / n_batches`), since these are logged every batch, not
  every epoch.
- Reward scores, per training epoch: `reward{1,2,3}_loss` — whichever of
  the three exist for a given run (a 2-reward run only has
  reward1/reward2), each a `(n_epochs + 1,)` array.

The two loss logs are 6-8 MB each and live in the git-ignored
`data/reward_weight_sweep_8x8_3reward/` tree. The two columns above, plus the
reward arrays in full, are baked into the tracked
`data/derived/<run>_losses.npz`, which this script prefers — so the Fig 6.10
run needs no download. A `--run-prefix` naming any other run (your own
training, say) falls through to the raw logs unchanged.

Example (the bundled 2-reward SatAndArea run)::

    python loss_curves.py --run-prefix \\
        "data/reward_weight_sweep_8x8_3reward/uniform_3c/multiRWGAN_R-SatGCN_300e_L0.2-d0.05_beta10_SatAndArea_100-0_date08072021_"
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from noc_data import DERIVED_DIR, derived_for  # noqa: E402

HERE = Path(__file__).resolve().parent
REWARD_LABELS = {1: "R-Sat", 2: "R-Pow", 3: "R-Area"}


def _load(prefix: str, suffix: str):
    path = Path(f"{prefix}{suffix}")
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def _load_derived(prefix: str) -> dict | None:
    """The baked ``derived/<run>_losses.npz`` for this prefix, if it ships with the repo.

    The artifact holds the two loss columns this figure plots (the raw
    ``(n_batches, 4)`` arrays are 6-8 MB each and git-ignored) plus the reward
    channels in full.
    """
    stripped = Path(prefix.rstrip("_"))
    path = derived_for(stripped, "_losses.npz")
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as z:
        return {
            "g_series": z["g_loss"],
            "d_series": z["d_loss"],
            "rewards": {int(i): z[f"reward{int(i)}"] for i in z["rewards_present"]},
        }


def require_run(prefix: str) -> dict:
    """The run's plotted series: the raw logs at ``prefix`` if present, else the baked artifact.

    ``--run-prefix`` is always given explicitly, so an existing raw log wins —
    the same "the path you named is the path you get" rule the other scripts'
    ``--…-path`` overrides follow. On a plain clone, where the raw tree is not
    there, the baked artifact for that run answers instead.
    """
    generator_loss = _load(prefix, "generator_loss")
    discriminator_loss = _load(prefix, "discriminator_loss")
    if generator_loss is None or discriminator_loss is None:
        derived = _load_derived(prefix)
        if derived is not None:
            return derived
        raise FileNotFoundError(
            f"No generator_loss/discriminator_loss at prefix {prefix!r}, and no baked "
            f"losses for that run under {DERIVED_DIR}/. Pass --run-prefix pointing at "
            "one of the runs under data/reward_weight_sweep_8x8_3reward/ (git-ignored — "
            "see data/README.md) or at your own training run; see this module's "
            "docstring for an example."
        )
    rewards = {}
    for i in (1, 2, 3):
        r = _load(prefix, f"reward{i}_loss")
        if r is not None:
            rewards[i] = np.asarray(r)
    # Column 0 of the generator log and column 2 of the critic log are the two
    # series Fig 6.10 draws; the other six are not plotted.
    return {
        "g_series": np.asarray(generator_loss)[:, 0],
        "d_series": np.asarray(discriminator_loss)[:, 2],
        "rewards": rewards,
    }


def plot_losses(run: dict, output_dir: Path, name: str) -> Path:
    g_loss = run["g_series"]
    d_loss = run["d_series"]
    # rewardN_loss is logged once per epoch (n_epochs + 1 values); g/d loss
    # is logged once per training batch — recover n_epochs from whichever
    # reward channel exists so the x-axis lines up in epoch units.
    n_epochs = len(next(iter(run["rewards"].values()))) - 1 if run["rewards"] else len(g_loss)
    x = np.arange(0, n_epochs, n_epochs / len(g_loss))[: len(g_loss)]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(x, g_loss, label="G loss", color="b")
    ax.plot(x, d_loss, label="C loss", color="r")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss value")
    ax.set_title("Generator / critic loss (Fig 6.10)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path = output_dir / f"{name}_losses.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_rewards(run: dict, output_dir: Path, name: str) -> Path | None:
    if not run["rewards"]:
        return None
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = {1: "orange", 2: "g", 3: "b"}
    for i, scores in sorted(run["rewards"].items()):
        ax.plot(scores[:-1], label=REWARD_LABELS[i], color=colors[i])
    ax.set_xlabel("epoch")
    ax.set_ylabel("score")
    ax.set_title("Reward scores over training (Fig 6.10)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path = output_dir / f"{name}_rewards.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--run-prefix",
        required=True,
        help="Shared filename prefix before 'generator_loss'/'discriminator_loss'/"
        "'reward{1,2,3}_loss', e.g. a path under "
        "data/reward_weight_sweep_8x8_3reward/<traffic>_3c/.",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Base name for output files (default: derived from --run-prefix).",
    )
    parser.add_argument("--output-dir", default=str(HERE / "output"))
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    run = require_run(args.run_prefix)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    name = args.name or Path(args.run_prefix.rstrip("_")).name

    loss_path = plot_losses(run, output_dir, name)
    print(f"wrote {loss_path}")
    reward_path = plot_rewards(run, output_dir, name)
    if reward_path:
        print(f"wrote {reward_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
