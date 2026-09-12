#!/usr/bin/env python3
"""Reproduce the DATE2022 router-rate-over-training curve (``picture_DATE2022_tauxRouteurs``).

A single line plot: the mean fraction of the 64 routers assigned to each of the
3 classes (Big / Medium / Small), per training epoch, averaged over a
fixed-noise generator sample batch. The DATE2022 notebook renders it five times
with different styling and saves each — this script does the same:

    BDtime3     colour (r/g/b), figsize 8.75x3
    BDtimeBW    greyscale, thick lines, figsize 7.5x2
    BDtimeBWr   red-scale
    BDtimeBW2   yellow/teal/blue
    BDtimeBW3   yellow/teal/blue (alt shades)

Input: one training run's per-epoch generator sample batches. By default the
bundled ``figures/data/derived/*_classes.npz`` for the uniform 3-reward run —
its per-router class indices, which is all this figure reduces the batches to.
The raw ``*_historyFake`` pickle it was baked from stays git-ignored; point
``--history-path`` at a copy to read one directly::

    python taux_routeurs_date2022.py --style all
    python taux_routeurs_date2022.py --style BDtimeBW --history-path <other run>_historyFake
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from noc_data import class_fraction_over_epochs, load_history, resolve_history  # noqa: E402

HERE = Path(__file__).resolve().parent
CLASS_LABELS = ("Big", "Medium", "Small")

# The whole reward_weight_sweep_8x8_3reward/ tree is git-ignored (>500 MB of
# *_historyFake pickles). The per-router class indices this figure needs are
# baked out of it into the tracked data/derived/*_classes.npz, which
# noc_data.resolve_history prefers -- so the default run needs no download.
_RUN = (
    "reward_weight_sweep_8x8_3reward/uniform_3c/"
    "multiRWGAN_R-SatGNN_R-PowGNN_R-AreaCNN_300e_L0.2-d0.05_beta10_"
    "SatAndPowAndArea_50-10-40_date16072021_historyFake"
)
DEFAULT_BUNDLE = HERE / "data" / _RUN

# Legend kwargs shared by every style except BDtime3 (verbatim from the notebook).
_BW_LEGEND = dict(
    fontsize=20, loc="upper left", bbox_to_anchor=(1, 0.78, 0.3, 0.3), labelspacing=0.9
)

STYLES = {
    "BDtime3": dict(
        figsize=(8.75, 3), colors=("r", "b", "g"), lw=None, fs=16,
        legend_kw=dict(fontsize=16, loc="upper left"),
    ),
    "BDtimeBW": dict(
        figsize=(7.5, 2), colors=((0, 0, 0), (0.4, 0.4, 0.4), (0.7, 0.7, 0.7)), lw=3, fs=20,
        legend_kw=_BW_LEGEND,
    ),
    "BDtimeBWr": dict(
        figsize=(7.5, 2), colors=((0.5, 0, 0), (1, 0.3, 0.3), (1, 0.8, 0.8)), lw=2.5, fs=20,
        legend_kw=_BW_LEGEND,
    ),
    "BDtimeBW2": dict(
        figsize=(7.5, 2), colors=((0.95, 0.9, 0.1), (0.1, 0.7, 0.5), (0, 0, 0.7)), lw=3, fs=20,
        legend_kw=_BW_LEGEND,
    ),
    "BDtimeBW3": dict(
        figsize=(7.5, 2), colors=((0.9, 0.85, 0.1), (0.1, 0.6, 0.4), (0, 0, 0.7)), lw=3, fs=20,
        legend_kw=_BW_LEGEND,
    ),
}


def plot_style(proportions, name: str, style: dict, output_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=style["figsize"])
    for k, label in enumerate(CLASS_LABELS):
        kw = {"label": label, "color": style["colors"][k]}
        if style["lw"] is not None:
            kw["linewidth"] = style["lw"]
        ax.plot(proportions[k, :-1], **kw)  # notebook drops the last logged epoch
    ax.legend(**style["legend_kw"])
    ax.grid(True)
    fs = style["fs"]
    ax.set_xlabel("epochs", fontsize=fs)
    ax.set_ylabel("router rates", fontsize=fs)
    ax.tick_params(labelsize=fs)
    fig.tight_layout()
    out_path = output_dir / f"{name}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--style", choices=("all", *STYLES), default="all")
    parser.add_argument(
        "--history-path", default=None,
        help="A raw *_historyFake pickle to read instead of the bundled "
        "data/derived/*_classes.npz. Default run: the uniform 3-reward "
        "SatAndPowAndArea 50-10-40 one.",
    )
    parser.add_argument("--mesh-rows", type=int, default=8)
    parser.add_argument("--mesh-cols", type=int, default=8)
    parser.add_argument("--nb-classes", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--output-dir", default=str(HERE / "output"))
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    history_path = resolve_history(
        args.history_path, DEFAULT_BUNDLE, what="taux-routeurs history",
    )
    history = load_history(history_path)
    n_routers = args.mesh_rows * args.mesh_cols
    proportions = class_fraction_over_epochs(
        history, n_routers, args.nb_classes, args.batch_size
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    names = list(STYLES) if args.style == "all" else [args.style]
    for name in names:
        out_path = plot_style(proportions, name, STYLES[name], output_dir)
        print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
