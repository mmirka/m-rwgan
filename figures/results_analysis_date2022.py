#!/usr/bin/env python3
"""Reproduce the DATE2022 results-analysis Pareto/scatter figures.

From ``picture_DATE2022_resultat_{uniform,hotspot,hotspot-area}_Analysis``. Each
plots the M-RWGAN weight-ratio sweep (13 configs: a 6-point SatAndArea GCN-CNN
run and a 7-point SatAndPow GCN-GCN run) against the homogeneous reference
topologies and, for the area figure, the binned dataset curve.

Figures (the notebooks' uncommented ``savefig`` names; this script prefixes the
traffic so the three notebooks' outputs don't overwrite each other):

    uniform      -> gene_uniSurf_isca4   (throughput vs power, SatAndPow half, error ellipses)
                    gene_uni_pareto_satar (throughput vs area, both halves + dataset curve)
    hotspot      -> geneSurf_hot_isca4   (as gene_uniSurf_isca4)
                    geneSurf_hot_isca    (throughput vs power, SatAndArea half)
                    gene_uni_pareto_satar
    hotspot-area -> geneSurf_hot_isca    (throughput vs power, SatAndPow half -- the
                                          hotspot-area notebook's variant)
                    gene_uni_pareto_satar

The reference and generated sweep data are git-tracked under ``figures/data/``,
and the dataset curve reads the bundled
``figures/data/derived/dataset_10k_*_summary.npz`` -- so nothing needs
downloading. To recompute that curve from the raw ~250-320 MB pickle instead,
fetch it per ``figures/data/README.md`` and pass ``--dataset-10k``::

    python results_analysis_date2022.py --traffic uniform
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse

from noc_data import load_data_ax_dataset, load_dataset_summary, min_max_normalize, summarize_samples  # noqa: E402

HERE = Path(__file__).resolve().parent

SWEEP_REWARDS = ("SatAndArea_GCN-CNN_", "SatAndPow_GCN-GCN_")
SWEEP_COEFS = ("100-0", "90-10", "70-30", "50-50", "30-70", "10-90", "0-100")
# The (SatAndArea, "100-0") combo is skipped -> 13 configs: [0:6] SatAndArea, [6:13] SatAndPow.

REF_NAME = ["Big", "Small", "Hetero1-SB", "Hetero2-SB", "Hetero3-SB",
            "Medium", "Hetero1-MB", "Hetero2-MB", "Hetero3-MB"]

# Per traffic: sweep-file prefix, the reference topology dirs, the 10k dataset
# stem (git-ignored, ~250-310 MB -- see figures/data/README.md) and the figures.
TRAFFIC = {
    "uniform": {
        "prefix": "uniform_",
        "refs": ("dataset_refNoC_uniform_hp112lp102_8x8", "dataset_refNoC_medium104_uni",
                 "dataset_refNoC_ISCAmed_uni"),
        "tenk": "dataset_10k_uniform_3c",
        "figures": ("gene_uniSurf_isca4", "gene_uni_pareto_satar"),
    },
    "hotspot": {
        "prefix": "hotspot30_",
        "refs": ("dataset_refNoC_hotspot30_hp112lp102_8x8", "dataset_refNoC_medium104_hot30",
                 "dataset_refNoC_ISCAmed_hot"),
        "tenk": "dataset_10k_hotspot30_3c",
        "figures": ("geneSurf_hot_isca4", "geneSurf_hot_isca", "gene_uni_pareto_satar"),
    },
    "hotspot-area": {
        "prefix": "hotspot30_",
        "refs": ("dataset_refNoC_hotspot30_hp112lp102_8x8", "dataset_refNoC_medium104_hot30",
                 "dataset_refNoC_ISCAmed_hot"),
        "tenk": "dataset_10k_hotspot30_3c",
        "figures": ("geneSurf_hot_isca", "gene_uni_pareto_satar"),
    },
}

# The "*_isca4" / "*_isca" scatter routine, parametrised. Offset lists are copied
# verbatim from each notebook cell (index 0..12, aligned with the 13 sweep configs).
ISCA_SPECS = {
    ("uniform", "gene_uniSurf_isca4"): dict(
        sl=slice(6, 13), figsize=(5, 6), fs=17, label="Generated T-P", axisbelow=True,
        legend_kw=dict(fontsize=17, labelspacing=0, handletextpad=0, borderpad=0),
        x_off_norm=[0, 0, 0, 0, 0, 0, 0.02, 0.025, 0.02, -0.22, -0.18, -0.17, 0.03],
        y_off_norm=[0, 0, 0, 0, 0, 0, 0.02, -0.06, 0.02, -0.02, 0.02, -0.04, -0.04],
        x_off_ref=[-0.13, 0.03, -0.12, 0.03, 0.03, 0.03, -0.41, -0.03, -0.41],
        y_off_ref=[-0.03, -0.04, 0.02, 0.01, -0.025, -0.06, 0.02, -0.06, -0.05],
    ),
    ("hotspot", "geneSurf_hot_isca4"): dict(
        sl=slice(6, 13), figsize=(5, 6), fs=17, label="Generated T-P", axisbelow=True,
        legend_kw=dict(fontsize=17, labelspacing=0, handletextpad=0, borderpad=0),
        x_off_norm=[0, 0, 0, 0, 0, 0, -0.19, -0.19, 0.03, -0.22, 0.03, 0.03, 0.03],
        y_off_norm=[0, 0, 0, 0, 0, 0, 0.02, 0.02, 0, -0.02, 0.01, -0.03, -0.03],
        x_off_ref=[-0.13, 0.03, -0.05, -0.15, -0.32, 0.02, -0.42, -0.42, -0.41],
        y_off_ref=[-0.03, -0.03, 0.05, 0.08, 0.03, -0.05, 0.025, -0.03, 0.01],
    ),
    ("hotspot", "geneSurf_hot_isca"): dict(
        sl=slice(0, 6), figsize=(7, 5), fs=15, label="Throughput-Area", axisbelow=False,
        legend_kw=dict(fontsize=15),
        x_off_norm=[0, 0, 0, 0, 0, 0, -0.04, -0.13, 0.02, -0.13, 0.01, 0.02, 0.02],
        y_off_norm=[0, 0, 0, 0, 0, 0, 0.03, 0.01, -0.02, 0.01, 0.01, -0.02, -0.02],
        x_off_ref=[-0.1, 0.02, -0.05, -0.1, 0.01, 0.01, -0.26, 0.01, -0.26],
        y_off_ref=[-0.03, -0.03, 0.03, 0.05, 0.01, -0.05, 0.01, 0.01, 0.01],
    ),
    ("hotspot-area", "geneSurf_hot_isca"): dict(
        sl=slice(6, 13), figsize=(7, 5), fs=15, label="Throughput-Power", axisbelow=False,
        legend_kw=dict(fontsize=15),
        x_off_norm=[0, 0, 0, 0, 0, 0, -0.04, -0.13, 0.02, -0.13, 0.01, 0.02, 0.02],
        y_off_norm=[0, 0, 0, 0, 0, 0, 0.03, 0.01, -0.02, 0.01, 0.01, -0.02, -0.02],
        x_off_ref=[-0.1, 0.02, -0.05, -0.1, 0.01, 0.01, -0.26, 0.01, -0.26],
        y_off_ref=[-0.03, -0.03, 0.03, 0.05, 0.01, -0.05, 0.01, 0.01, 0.01],
    ),
}

# gene_uni_pareto_satar (identical across the three notebooks).
SATAR_X_OFF = [0.001] * 9 + [-0.016] * 4
SATAR_Y_OFF = [-0.05, 0, -0.05, -0.05, -0.05, -0.05, -0.02, -0.01, -0.05, 0.01, 0.01, 0.01, 0.01]
SATAR_COLORS = ["b"] * 6 + ["g"] * 7


def _summ_concat(dirs) -> dict:
    parts = [summarize_samples(load_data_ax_dataset(str(p))) for p in dirs]
    return {m: np.concatenate([p[m] for p in parts]) for m in ("sat", "pow", "area")}


def load_sweep(prefix: str, generated_dir: Path):
    """13 configs -> per-config mean/std of sat/pow/area, plus the coef-string names."""
    names, out = [], {f"{m}_{s}": [] for m in ("sat", "pow", "area") for s in ("mean", "std")}
    for reward in SWEEP_REWARDS:
        for coef in SWEEP_COEFS:
            if reward == SWEEP_REWARDS[0] and coef == "100-0":
                continue
            path = generated_dir / f"{prefix}{reward}{coef}_dataset"
            summ = summarize_samples(load_data_ax_dataset(str(path)))
            names.append(coef)
            for m in ("sat", "pow", "area"):
                out[f"{m}_mean"].append(float(summ[m].mean()))
                out[f"{m}_std"].append(float(summ[m].std()))
    arrs = {k: np.asarray(v) for k, v in out.items()}
    arrs["names"] = names
    return arrs


def dataset_curve(summ: dict) -> dict:
    """Bin the 10k dataset summary by distinct (rounded) saturation value -> per-bin means.

    ``summ`` is ``{'sat', 'pow', 'area'}`` — either read straight from the
    bundled ``data/derived/dataset_10k_*_summary.npz`` or recomputed from the
    raw pickle when ``--dataset-10k`` names one.
    """
    curve = {"sat": [], "pow": [], "area": []}
    for value in sorted(set(summ["sat"])):
        mask = summ["sat"] == value
        for m in ("sat", "pow", "area"):
            curve[m].append(float(summ[m][mask].mean()))
    return {m: np.asarray(v) for m, v in curve.items()}


def plot_isca(sweep, refs, spec, out_path: Path) -> Path:
    mn_s, mx_s = refs["sat"].min(), refs["sat"].max()
    mn_p, mx_p = refs["pow"].min(), refs["pow"].max()
    sat_norm = min_max_normalize(sweep["sat_mean"], mn_s, mx_s)
    pow_norm = min_max_normalize(sweep["pow_mean"], mn_p, mx_p)
    sat_ref_norm = min_max_normalize(refs["sat"], mn_s, mx_s)
    pow_ref_norm = min_max_normalize(refs["pow"], mn_p, mx_p)
    sl = spec["sl"]

    fig, ax = plt.subplots(figsize=spec["figsize"])
    ax.errorbar(sat_norm[sl], pow_norm[sl], linestyle="None", marker="o", markersize=10,
                color="r", label=spec["label"])

    for i in range(sl.start, sl.stop):
        std_x = min_max_normalize(sweep["sat_std"][i] + mn_s, mn_s, mx_s)
        std_y = min_max_normalize(sweep["pow_std"][i] + mn_p, mn_p, mx_p)
        e = Ellipse((sat_norm[i], pow_norm[i]), 2 * std_x, 2 * std_y, angle=0)
        e.set_clip_box(ax.bbox)
        e.set_alpha(0.2)
        e.set_facecolor("r")
        ax.add_artist(e)
        ax.annotate(sweep["names"][i],
                    (sat_norm[i] + spec["x_off_norm"][i], pow_norm[i] + spec["y_off_norm"][i]),
                    fontsize=spec["fs"], color="r")

    ax.scatter(sat_ref_norm, pow_ref_norm, marker="+", s=100, color="b", label="Single Conf.")
    for i in range(len(sat_ref_norm)):
        ax.annotate(REF_NAME[i],
                    (sat_ref_norm[i] + spec["x_off_ref"][i], pow_ref_norm[i] + spec["y_off_ref"][i]),
                    fontsize=spec["fs"], color="b")

    ax.set_xlabel("Throughput", fontsize=spec["fs"])
    ax.set_ylabel("Power", fontsize=spec["fs"])
    ax.tick_params(labelsize=spec["fs"])
    if spec["axisbelow"]:
        ax.set_axisbelow(True)
    ax.grid(True)
    ax.legend(**spec["legend_kw"])

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_pareto_satar(sweep, curve, out_path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(sweep["sat_mean"][0:6], sweep["area_mean"][0:6], linestyle="None", marker="o",
                color="b", label="Sat-Area")
    ax.errorbar(sweep["sat_mean"][6:], sweep["area_mean"][6:], linestyle="None", marker="o",
                color="g", label="Sat-En")
    ax.errorbar(curve["sat"], curve["area"], linestyle="-", marker="^", color="k", label="dataset")
    for i in range(len(sweep["names"])):
        ax.annotate(sweep["names"][i],
                    (sweep["sat_mean"][i] + SATAR_X_OFF[i], sweep["area_mean"][i] + SATAR_Y_OFF[i]),
                    fontsize=15, color=SATAR_COLORS[i])
    ax.set_xlabel("Taux de saturation", fontsize=15)
    ax.set_ylabel("Surface normalisée", fontsize=15)
    ax.tick_params(labelsize=15)
    ax.grid(True)
    ax.legend(fontsize=15, title_fontsize=15)

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--traffic", required=True, choices=tuple(TRAFFIC))
    parser.add_argument(
        "--figure", default="active",
        choices=("active", "gene_uniSurf_isca4", "geneSurf_hot_isca4", "geneSurf_hot_isca",
                 "gene_uni_pareto_satar"),
        help="Which figure(s) to draw. 'active' = the ones the chosen notebook saves.",
    )
    parser.add_argument(
        "--dataset-10k", default=None,
        help="Raw data/dataset_10k_<traffic>_3c pickle to summarise instead of the "
        "bundled data/derived/*_summary.npz.",
    )
    parser.add_argument("--generated-dir", default=str(HERE / "data" / "generated" / "10-50"))
    parser.add_argument("--output-dir", default=str(HERE / "output"))
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = TRAFFIC[args.traffic]
    figures = cfg["figures"] if args.figure == "active" else (args.figure,)
    figures = [f for f in figures if f in cfg["figures"]]
    if not figures:
        raise SystemExit(f"--figure {args.figure!r} is not produced for --traffic {args.traffic}")

    ref_dir = HERE / "data" / "baseline_topology_dataset"
    refs = _summ_concat([ref_dir / name / "dataset" for name in cfg["refs"]])
    sweep = load_sweep(cfg["prefix"], Path(args.generated_dir))

    curve = None
    if "gene_uni_pareto_satar" in figures:
        curve = dataset_curve(load_dataset_summary(
            args.dataset_10k, HERE / "data" / cfg["tenk"],
            what=f"{args.traffic} dataset_10k",
        ))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for fig_name in figures:
        out_path = output_dir / f"{args.traffic}_{fig_name}.png"
        if fig_name == "gene_uni_pareto_satar":
            plot_pareto_satar(sweep, curve, out_path)
        else:
            plot_isca(sweep, refs, ISCA_SPECS[(args.traffic, fig_name)], out_path)
        print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
