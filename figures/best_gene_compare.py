#!/usr/bin/env python3
"""Reproduce the DATE2022 best-configuration comparison (``picture_DATE2022_bestGene``, ``gene_compare4``).

Two stacked panels sharing an injection-rate x-axis:

- **top**  -- latency vs injection rate
- **bottom** -- power vs injection rate

Each panel overlays five curves: the three homogeneous reference topologies
(Small / Medium / Big, dashed black) and two M-RWGAN-generated heterogeneous
configurations -- ``Conf. P`` (a low-power pick from the SatAndPow GCN-GCN 10-90
sweep) and ``Conf. T`` (a high-throughput pick from the SatAndPowAndArea
50-10-40 set), both solid red.

All inputs are bundled under ``figures/data/``::

    python best_gene_compare.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from noc_data import load_data_ax_dataset  # noqa: E402

HERE = Path(__file__).resolve().parent

DEFAULTS = {
    "ref_hot": "data/baseline_topology_dataset/dataset_refNoC_hotspot30_hp112lp102_8x8/dataset",
    "ref_hot_medium": "data/baseline_topology_dataset/dataset_refNoC_medium104_hot30/dataset",
    "confP": "data/generated/10-50/hotspot30_SatAndPow_GCN-GCN_10-90_dataset",
    "confT": "data/generated/hotspot30_SatAndPowAndArea_50-10-40_dataset",
}

# label -> (colour, linestyle, marker); order = plotting/legend order (matches the notebook).
SERIES_STYLE = {
    "Small": ("k", "--", "v"),
    "Medium": ("k", "--", "s"),
    "Big": ("k", "--", "^"),
    "Conf. P": ("r", "-", "v"),
    "Conf. T": ("r", "-", "^"),
}


def _pick(dataset_path: str, index: int, what: str):
    samples = load_data_ax_dataset(dataset_path)
    if not -len(samples) <= index < len(samples):
        raise IndexError(f"{what}: index {index} out of range for {len(samples)} samples in {dataset_path}")
    return samples[index]


def _plot_curve(ax, sample, attr: str, label: str):
    colour, ls, marker = SERIES_STYLE[label]
    curve = np.asarray(getattr(sample, attr))  # (2, n): row 0 = injection rate, row 1 = metric
    ax.plot(curve[0], curve[1], label=label, color=colour, linestyle=ls, marker=marker,
            linewidth=2, markersize=8)


def build_figure(series: dict, output_dir: Path) -> Path:
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, sharey=False, figsize=(11, 4))

    for label, sample in series.items():
        _plot_curve(ax1, sample, "latency", label)
    ax1.set_ylim((0, 450))
    ax1.set_xlim((0.01, 0.045))
    ax1.grid()
    ax1.set_ylabel("Latency (ms)", fontsize=16)
    ax1.tick_params(labelsize=16)

    for label, sample in series.items():
        _plot_curve(ax2, sample, "total_power", label)
    ax2.set_xlim((0.01, 0.045))
    ax2.grid()
    ax2.set_xlabel("Injection rate (flits/cycle)", fontsize=16)
    ax2.set_ylabel("Power (mW)", fontsize=16)
    ax2.tick_params(labelsize=16)

    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(
        handles, labels, ncol=5, loc="center", bbox_to_anchor=(0.35, 1.01, 0.3, 0.3),
        labelspacing=0, handlelength=2, columnspacing=0.5, handletextpad=0.2, fontsize=16,
    )
    plt.subplots_adjust(wspace=0, hspace=0.1)

    out_path = output_dir / "gene_compare4.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--ref-hot-dataset", default=str(HERE / DEFAULTS["ref_hot"]))
    parser.add_argument("--ref-hot-medium-dataset", default=str(HERE / DEFAULTS["ref_hot_medium"]))
    parser.add_argument("--confP-dataset", default=str(HERE / DEFAULTS["confP"]))
    parser.add_argument("--confP-index", type=int, default=18)
    parser.add_argument("--confT-dataset", default=str(HERE / DEFAULTS["confT"]))
    parser.add_argument("--confT-index", type=int, default=67)
    parser.add_argument("--output-dir", default=str(HERE / "output"))
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    ref_hot = load_data_ax_dataset(args.ref_hot_dataset)
    ref_hot_medium = load_data_ax_dataset(args.ref_hot_medium_dataset)

    series = {
        "Small": ref_hot[1],
        "Medium": ref_hot_medium[0],
        "Big": ref_hot[0],
        "Conf. P": _pick(args.confP_dataset, args.confP_index, "Conf. P"),
        "Conf. T": _pick(args.confT_dataset, args.confT_index, "Conf. T"),
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = build_figure(series, output_dir)
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
