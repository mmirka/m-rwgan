#!/usr/bin/env python3
"""Reproduce the DATE2022 box-and-whisker training-evolution figures (``picture_DATE2022_moustache``).

For a given traffic pattern (uniform / hotspot) and reward weighting (T100 =
pure throughput, or T50P10A40 = the 3-reward mix), the distribution of three
normalized metrics -- Throughput, Power, Area -- is drawn as a box per training
epoch snapshot {0, 100, 200, 300}. Each metric is min-max-normalized against the
homogeneous reference topologies for that traffic. Boxes use a 5th/95th-style
whisker (the exact percentile pair varies per panel, matching the notebook) and
a red mean line.

Active figures (the notebook's three uncommented ``savefig`` calls):

    moustache_hotzoom2_moy3   hotspot, T50P10A40 only (1 panel)
    moustache_uni3            uniform, T100 + T50P10A40 (2 panels)
    moustache_all             uniform + hotspot, both weightings (4 panels)

Epoch 0 is the full ``dataset_10k_{uniform,hotspot30}_3c``, read from the
bundled ``figures/data/derived/dataset_10k_*_summary.npz`` (the raw ~250-320 MB
pickles stay git-ignored; pass ``--epoch0-uniform`` / ``--epoch0-hotspot`` to
summarise one directly). Epochs 100/200 use the tracked ``data/moustache/``
sets; epoch 300 uses the tracked ``data/generated/`` sets::

    python moustache_boxplots.py --figure all
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.cbook as cbook
import matplotlib.pyplot as plt
import numpy as np

from noc_data import load_data_ax_dataset, load_dataset_summary, min_max_normalize, summarize_samples  # noqa: E402

HERE = Path(__file__).resolve().parent

# traffic key -> (reference dataset dir name, generated-file infix, 10k dataset
# stem). The "tenk" sets are the git-ignored ~250-310 MB files -- see
# figures/data/README.md.
TRAFFIC = {
    "uni": {
        "ref": "dataset_refNoC_uniform_hp112lp102_8x8",
        "gen": "uniform",
        "tenk": "dataset_10k_uniform_3c",
    },
    "hot": {
        "ref": "dataset_refNoC_hotspot30_hp112lp102_8x8",
        "gen": "hotspot30",
        "tenk": "dataset_10k_hotspot30_3c",
    },
}
METRICS = ("sat", "pow", "area")

# Per-metric box styling (verbatim from the notebook): Throughput / Power / Area.
TRIO_STYLE = (
    {"facecolor": "k", "cap_ls": "-"},
    {"facecolor": "grey", "cap_ls": "--"},
    {"facecolor": "white", "cap_ls": "--"},
)
FLIERPROPS = dict(marker="+", markerfacecolor="none", markersize=10, markeredgecolor="k")
# The notebook uses linestyle "-." here but linewidth 0 hides the median anyway;
# a dashed style with zero width trips newer matplotlib, so keep it solid.
MEDIANPROPS = dict(linestyle="-", linewidth=0, color="r")
MEANLINEPROPS = dict(linestyle="-", linewidth=3, color="r")
WHISKERPROPS = dict(linestyle="--", linewidth=2, color="k")

POS_12 = ([1, 6, 11, 16], [2, 7, 12, 17], [3, 8, 13, 18])  # hotzoom2 / uni3
POS_ALL = ([1, 5, 9, 13], [2, 6, 10, 14], [3, 7, 11, 15])   # moustache_all (NB typo [3,7,10,15] fixed)

# Which (traffic, weight, percentile-pair) each panel of each figure shows.
FIGURES = {
    "moustache_hotzoom2_moy3": {
        "figsize": (8, 4), "naxes": 1, "positions": POS_12, "xticks": [2, 7, 12, 17],
        "xlim": (0, 19), "label_fs": 18, "legend_anchor": (0.35, 1.1, 0.3, 0.3),
        "panels": [("hot", "501040", (5, 95), "T$_{50}$P$_{10}$A$_{40}$")],
    },
    "moustache_uni3": {
        "figsize": (12, 3), "naxes": 2, "positions": POS_12, "xticks": [2, 7, 12, 17],
        "xlim": (0, 19), "label_fs": 18, "legend_anchor": (0.9, 1.1, 0.3, 0.3),
        "panels": [
            ("uni", "100", (1.5, 98.5), "T$_{100}$"),
            ("uni", "501040", (1.5, 98.5), "T$_{50}$P$_{10}$A$_{40}$"),
        ],
    },
    "moustache_all": {
        "figsize": (12, 5), "naxes": 4, "positions": POS_ALL, "xticks": [2, 6, 10, 14],
        "xlim": (0, 16), "label_fs": 16, "legend_anchor": (1.98, -0.42, 0.3, 0.3),
        "panels": [
            ("uni", "100", (1.5, 98.5), "T$_{100}$"),
            ("uni", "501040", (1.5, 98.5), "T$_{50}$P$_{10}$A$_{40}$"),
            ("hot", "100", (5, 95), "T$_{100}$"),
            ("hot", "501040", (2.5, 97.5), "T$_{50}$P$_{10}$A$_{40}$"),
        ],
    },
}


def _bxp_stats(x, plo: float, phi: float) -> dict:
    x = np.asarray(x)
    stat = cbook.boxplot_stats(x)[0]
    stat["q1"], stat["q3"] = np.percentile(x, [plo, phi])
    return stat


def _draw_panel(ax, series4: dict, plo: float, phi: float, positions) -> list:
    """Draw the 3 metric trios (4 epoch boxes each) on `ax`; return the 3 bxp dicts."""
    bps = []
    for ti, metric in enumerate(METRICS):
        stats = [_bxp_stats(x, plo, phi) for x in series4[metric]]
        st = TRIO_STYLE[ti]
        bp = ax.bxp(
            stats, positions=positions[ti], widths=0.6, patch_artist=True,
            meanline=True, showmeans=True,
            boxprops=dict(linestyle="-", linewidth=2, edgecolor="k", facecolor=st["facecolor"]),
            flierprops=FLIERPROPS, medianprops=MEDIANPROPS, meanprops=MEANLINEPROPS,
            capprops=dict(linestyle=st["cap_ls"], linewidth=2, color="k"),
            whiskerprops=WHISKERPROPS,
        )
        bps.append(bp)
    return bps


class DataResolver:
    """Resolves + caches the normalized per-metric arrays each panel needs."""

    def __init__(self, args):
        self.args = args
        self._ref_bounds = {}   # traffic -> {metric: (min, max)}
        self._norm = {}         # (traffic, weight, epoch) -> {metric: np.ndarray}


    def ref_bounds(self, traffic: str) -> dict:
        if traffic not in self._ref_bounds:
            path = HERE / "data" / "baseline_topology_dataset" / TRAFFIC[traffic]["ref"] / "dataset"
            summ = summarize_samples(load_data_ax_dataset(str(path)))
            self._ref_bounds[traffic] = {m: (summ[m].min(), summ[m].max()) for m in METRICS}
        return self._ref_bounds[traffic]

    def _summary(self, traffic: str, weight: str, epoch: int) -> dict:
        """The {sat, pow, area} arrays for one panel's dataset.

        Epoch 0 is the whole 10k training set, which is why it is the one
        panel served from ``data/derived/`` rather than from a tracked
        pickle: the summary is all this figure takes from it.
        """
        if epoch == 0:
            return load_dataset_summary(
                self.args.epoch0_uniform if traffic == "uni" else self.args.epoch0_hotspot,
                HERE / "data" / TRAFFIC[traffic]["tenk"],
                what=f"{traffic} epoch-0 dataset",
            )
        return summarize_samples(load_data_ax_dataset(str(self._dataset_path(traffic, weight, epoch))))

    def _dataset_path(self, traffic: str, weight: str, epoch: int) -> Path:
        gen = TRAFFIC[traffic]["gen"]
        if epoch in (100, 200):
            return Path(self.args.moustache_dir) / f"dataset_{traffic}{weight}_e{epoch}"
        # epoch 300: the re-simulated final generator output (== bundled generated/ files)
        if weight == "100":
            return HERE / "data" / "generated" / "10-50" / f"{gen}_SatAndPow_GCN-GCN_100-0_dataset"
        return HERE / "data" / "generated" / f"{gen}_SatAndPowAndArea_50-10-40_dataset"

    def normalized(self, traffic: str, weight: str, epoch: int) -> dict:
        key = (traffic, weight, epoch)
        if key not in self._norm:
            summ = self._summary(traffic, weight, epoch)
            bounds = self.ref_bounds(traffic)
            self._norm[key] = {
                m: min_max_normalize(summ[m], bounds[m][0], bounds[m][1]) for m in METRICS
            }
        return self._norm[key]

    def series4(self, traffic: str, weight: str) -> dict:
        """{metric: [epoch0, epoch100, epoch200, epoch300]} of normalized arrays."""
        per_epoch = {e: self.normalized(traffic, weight, e) for e in (0, 100, 200, 300)}
        return {m: [per_epoch[e][m] for e in (0, 100, 200, 300)] for m in METRICS}


def build_figure(name: str, resolver: DataResolver, output_dir: Path) -> Path:
    cfg = FIGURES[name]
    fig, axes = plt.subplots(1, cfg["naxes"], sharex=False, sharey=True, figsize=cfg["figsize"])
    axes = np.atleast_1d(axes)
    label_fs = cfg["label_fs"]

    bps_first = None
    for ax, (traffic, weight, (plo, phi), title) in zip(axes, cfg["panels"]):
        series4 = resolver.series4(traffic, weight)
        bps = _draw_panel(ax, series4, plo, phi, cfg["positions"])
        if bps_first is None:
            bps_first = bps
        ax.set_xlim(*cfg["xlim"])
        ax.set_xticks(cfg["xticks"])
        ax.set_xticklabels(["0", "100", "200", "300"])
        ax.tick_params(labelsize=label_fs)
        ax.set_xlabel("Epochs", fontsize=16 if name == "moustache_all" else label_fs)
        ax.grid()
        ax.set_title(title, fontsize=23)
    axes[0].set_ylabel(
        "Normalized value", fontsize=18, labelpad=0 if name == "moustache_all" else None
    )

    # Legend (proxy handles: hm = red mean line, hl = spacer used only by moustache_all).
    hm, = axes[0].plot([1, 1], "r-", linewidth=3)
    handles = [b["boxes"][0] for b in bps_first] + [hm]
    labels = ["Throughput", "Power", "Area", "Mean"]
    if name == "moustache_all":
        hl, = axes[0].plot([], marker="", ls="")
        handles = [hl] + handles
        labels = ["legend:"] + labels
        hl.set_visible(False)
    axes[0].legend(
        handles, labels, fontsize=16, ncol=5, loc="center", bbox_to_anchor=cfg["legend_anchor"],
        labelspacing=1, handlelength=0.5, columnspacing=3, handletextpad=0.2,
    )
    hm.set_visible(False)

    if name == "moustache_all":
        text_kwargs = dict(ha="center", va="center", fontsize=20, color="k")
        axes[0].text(17, 1.25, "Uniform", **text_kwargs)
        axes[0].text(51, 1.25, "Hotspot", **text_kwargs)
        for xy, xytext in (((0, 1.12), (2.08, 1.12)), ((2.16, 1.12), (4.24, 1.12))):
            axes[0].annotate(
                "", xy=xy, xycoords="axes fraction", xytext=xytext, textcoords="axes fraction",
                arrowprops=dict(arrowstyle="-", connectionstyle="arc3", lw=1),
            )

    plt.subplots_adjust(wspace=0.08 if name == "moustache_all" else 0.1)
    out_path = output_dir / f"{name}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--figure", choices=("all", *FIGURES), default="all")
    parser.add_argument(
        "--epoch0-uniform", default=None,
        help="Raw dataset_10k_uniform_3c pickle to summarise instead of the bundled "
        "data/derived/*_summary.npz.",
    )
    parser.add_argument(
        "--epoch0-hotspot", default=None,
        help="Raw dataset_10k_hotspot30_3c pickle to summarise instead of the bundled "
        "data/derived/*_summary.npz.",
    )
    parser.add_argument("--moustache-dir", default=str(HERE / "data" / "moustache"))
    parser.add_argument("--output-dir", default=str(HERE / "output"))
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    resolver = DataResolver(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    names = list(FIGURES) if args.figure == "all" else [args.figure]
    for name in names:
        out_path = build_figure(name, resolver, output_dir)
        print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
