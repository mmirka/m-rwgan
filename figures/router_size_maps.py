#!/usr/bin/env python3
"""Reproduce the DATE2022 5-panel router-size maps (``picture_DATE2022_4uniform`` / ``_4hotspot30``).

One row of five 8x8 heatmaps:

    [ Traffic | Origin | T100 | T80P10A10 | T50P10A40 ]

- **Traffic**: the per-router traffic-load map for the chosen pattern.
- **Origin / T...**: a per-router "size score" = weighted sum of the class
  fractions the generator produces (Big=1, Medium=0.2, Small=0), at a chosen
  training epoch, for three reward-weight settings (pure throughput T100, and
  the two 3-reward mixes). *Origin* is the earliest logged epoch of the T100
  run.

Inputs per traffic pattern: the traffic map under ``data/traffic_maps/``
(git-tracked) plus three training runs' per-epoch generator batches, read from
the bundled ``data/derived/*_classes.npz``. The raw ``*_historyFake`` pickles
they were baked from stay git-ignored; the three ``--history-*`` options point
at copies of those directly::

    python router_size_maps.py --traffic hotspot
    python router_size_maps.py --traffic uniform
"""
from __future__ import annotations

import argparse
import pickle
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from noc_data import class_score_map, load_history, per_router_class_fraction, resolve_history  # noqa: E402

HERE = Path(__file__).resolve().parent

# All the *_historyFake pickles below live under figures/data/<_SWEEP>/, which is
# git-ignored in full (>500 MB; uniform's T100 run alone is ~493 MB). The
# per-router class indices these panels need are baked out of them into the
# tracked data/derived/*_classes.npz, which noc_data.resolve_history prefers.
_SWEEP = "reward_weight_sweep_8x8_3reward"

# Per traffic: (history basename for sat100 / 801010 / 501040, traffic-map stem,
# default epoch indices [Origin, T100, T80P10A10, T50P10A40], output basename).
TRAFFIC = {
    "uniform": {
        "subdir": "uniform_3c",
        "sat100": "multiRWGAN_R-SatGCN_300e_L0.2-d0.05_beta10_SatAndArea_100-0_date08072021_historyFake",
        "w801010": "multiRWGAN_R-SatGNN_R-PowGNN_R-AreaCNN_300e_L0.2-d0.05_beta10_SatAndPowAndArea_80-10-10_date09092021_historyFake",
        "w501040": "multiRWGAN_R-SatGNN_R-PowGNN_R-AreaCNN_300e_L0.2-d0.05_beta10_SatAndPowAndArea_50-10-40_date16072021_historyFake",
        "img": "uni_img",
        "norm": "norm_uni_traf",
        "epochs": (0, 283, 300, 300),
        "out": "4uni_t2",
    },
    "hotspot": {
        "subdir": "hotspot30_3c",
        "sat100": "multiRWGAN_R-SatGNN_R-AreaCNN_300e_L0.2-d0.05_beta10_SatAndArea_100-0_date16072021_historyFake",
        "w801010": "multiRWGAN_R-SatGNN_R-PowGNN_R-AreaCNN_300e_L0.2-d0.05_beta10_SatAndPowAndArea_80-10-10_date09092021_historyFake",
        "w501040": "multiRWGAN_R-SatGNN_R-PowGNN_R-AreaCNN_300e_L0.2-d0.05_beta10_SatAndPowAndArea_50-10-40_date08092021_historyFake",
        "img": "hot_img",
        "norm": "norm_hot_traf",
        "epochs": (0, 300, 300, 300),
        "out": "4hot_t2",
    },
}

PANEL_TITLES = ("Traffic", "Origin", "T$_{100}$", "T$_{80}$P$_{10}$A$_{10}$", "T$_{50}$P$_{10}$A$_{40}$")


def _load_history(explicit, basename: str, subdir: str, what: str):
    """One run's per-epoch batches: the bundled derived artifact, or a raw pickle."""
    return load_history(resolve_history(
        explicit, HERE / "data" / _SWEEP / subdir / basename, what=what,
    ))


def _load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def build_figure(
    img, norm_traf, score_maps, titles, output_dir: Path, out_name: str
) -> Path:
    vlo, vhi = float(norm_traf[0].min()), float(norm_traf[0].max())

    fig = plt.figure(figsize=(15, 5), facecolor="w")
    im_traffic = im_last = None
    for k in range(1, 6):
        ax = fig.add_subplot(1, 5, k)
        if k == 1:
            im_traffic = ax.imshow(
                np.asarray(img) * 100, vmin=vlo * 100, vmax=vhi * 100, cmap="inferno"
            )
            ax.set_title(titles[0], weight="normal", size=30)
        else:
            im_last = ax.imshow(score_maps[k - 2], vmin=0, vmax=1, cmap="viridis")
            ax.set_title(titles[k - 1], weight="normal", size=30 if k == 2 else 35)
        ax.tick_params(labelsize=30)
        ax.locator_params(axis="y", nbins=4)

    # Left colorbar (traffic load), placed by hand exactly as the notebook does.
    cbar1_ax = fig.add_axes([-0.015, 0.25, 0.02, 0.5])
    cb1 = fig.colorbar(im_traffic, cax=cbar1_ax)
    cb1.set_label("Load (%)", weight="normal", size=30, rotation=90, labelpad=-100)
    cb1.ax.tick_params(labelsize=30)
    cbar1_ax.yaxis.set_ticks_position("left")

    # Right colorbar (router size score).
    cbar_ax = fig.add_axes([1, 0.25, 0.02, 0.5])
    cb = fig.colorbar(im_last, cax=cbar_ax)
    cb.set_label("router's size", weight="normal", size=30, rotation=-90, labelpad=30)
    cb.ax.tick_params(labelsize=30)

    with warnings.catch_warnings():  # tight_layout + hand-placed axes -> harmless warning
        warnings.simplefilter("ignore")
        fig.tight_layout()

    out_path = output_dir / f"{out_name}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--traffic", required=True, choices=tuple(TRAFFIC))
    parser.add_argument("--history-sat100", default=None, help="A raw *_historyFake pickle for the T100 panel (default: the bundled derived artifact).")
    parser.add_argument("--history-801010", default=None, help="A raw *_historyFake pickle for the T80P10A10 panel.")
    parser.add_argument("--history-501040", default=None, help="A raw *_historyFake pickle for the T50P10A40 panel.")
    parser.add_argument(
        "--traffic-map-dir", default=str(HERE / "data" / "traffic_maps"),
        help="Directory holding {uni,hot}_img and norm_{uni,hot}_traf.",
    )
    parser.add_argument(
        "--epochs", type=int, nargs=4, default=None, metavar=("ORIGIN", "T100", "T80", "T50"),
        help="Epoch index per score panel. Default: 0 283 300 300 (uniform) / 0 300 300 300 (hotspot).",
    )
    parser.add_argument("--mesh-rows", type=int, default=8)
    parser.add_argument("--mesh-cols", type=int, default=8)
    parser.add_argument("--nb-classes", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--output-dir", default=str(HERE / "output"))
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = TRAFFIC[args.traffic]
    epochs = tuple(args.epochs) if args.epochs is not None else cfg["epochs"]
    n_routers = args.mesh_rows * args.mesh_cols

    hist_sat100 = _load_history(
        args.history_sat100, cfg["sat100"], cfg["subdir"],
        f"{args.traffic} T100 history",
    )
    hist_801010 = _load_history(
        args.history_801010, cfg["w801010"], cfg["subdir"],
        f"{args.traffic} T80P10A10 history",
    )
    hist_501040 = _load_history(
        args.history_501040, cfg["w501040"], cfg["subdir"],
        f"{args.traffic} T50P10A40 history",
    )
    for label, hist in (("T100", hist_sat100), ("T80P10A10", hist_801010), ("T50P10A40", hist_501040)):
        if len(hist) <= max(epochs):
            raise ValueError(f"{label} history has {len(hist)} epochs; need > {max(epochs)}")

    def score(hist, epoch):
        frac = per_router_class_fraction(hist[epoch], n_routers, args.nb_classes, args.batch_size)
        return class_score_map(frac, args.mesh_rows, args.mesh_cols)

    score_maps = [
        score(hist_sat100, epochs[0]),  # Origin
        score(hist_sat100, epochs[1]),  # T100
        score(hist_801010, epochs[2]),  # T80P10A10
        score(hist_501040, epochs[3]),  # T50P10A40
    ]

    map_dir = Path(args.traffic_map_dir)
    img = _load_pickle(map_dir / cfg["img"])
    norm_traf = np.asarray(_load_pickle(map_dir / cfg["norm"]))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = build_figure(img, norm_traf, score_maps, PANEL_TITLES, output_dir, cfg["out"])
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
