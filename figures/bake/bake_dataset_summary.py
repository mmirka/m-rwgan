#!/usr/bin/env python3
"""Bake a ``dataset_10k_*`` pickle down to the three arrays the figures plot.

The 10k-sample simulated datasets (``dataset_10k_uniform_3c``, 323 MB;
``dataset_10k_hotspot30_3c``, 253 MB) are pickled lists of 10,000 ``Data_AX``
objects, each carrying a full latency/power/area curve. Every figure that
reads them —

    pareto_figures.py            the "dataset" frontier and its mean
    results_analysis_date2022.py the gene_uni_pareto_satar dataset curve
    moustache_boxplots.py        the epoch-0 box

— consumes them through exactly one call, ``noc_data.summarize_samples``,
which reduces each sample to three numbers: saturation point, total power,
normalized area. So the sufficient statistic is a ``(10000,)`` triple, and
everything else in the pickle is discarded.

float64 is kept rather than narrowed: the whole artifact is ~250 KB either
way, and float64 makes the figures rebuild bit-identically from the derived
data, which is what ``bake/verify.sh`` checks.

    python bake/bake_dataset_summary.py                    # both datasets
    python bake/bake_dataset_summary.py dataset_10k_uniform_3c
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bake.bake_common import DATA_DIR, report, require, write_artifact  # noqa: E402
from noc_data import derived_rel, load_data_ax_dataset, summarize_samples  # noqa: E402

DATASETS = ("dataset_10k_uniform_3c", "dataset_10k_hotspot30_3c")


def bake(stem: str) -> Path:
    src = require(DATA_DIR / stem, f"{stem} (10k simulated dataset)")
    summ = summarize_samples(load_data_ax_dataset(str(src)))
    out = write_artifact(
        derived_rel(src, "_summary.npz"), src,
        {m: summ[m].astype("float64") for m in ("sat", "pow", "area")},
        baked_by="bake/bake_dataset_summary.py",
    )
    print(f"{stem}: {len(summ['sat'])} samples")
    report(out, src)
    return out


def main(argv=None) -> int:
    names = list(argv or sys.argv[1:]) or list(DATASETS)
    for name in names:
        bake(name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
