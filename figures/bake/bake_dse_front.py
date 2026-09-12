#!/usr/bin/env python3
"""Bake the 12-router/3-class design-space enumeration down to its Pareto front.

``data/dse_12r3c/dataset_all_{X,sat,pow}`` is the full 3^12 = 531,441-NoC
enumeration of the smaller 12-router mesh (59 MB). ``igd_compare.py`` uses it
for one thing only: ``noc_data.load_true_pareto_front``, which normalizes
saturation/power/area over the whole set and then, per distinct saturation
value, keeps the minimum power and minimum area — the marginal "true Pareto
front" the generated NoCs are measured against.

That front is three short arrays. The 531,441 rows behind it never reach the
plot, so the enumeration is the input to the reduction, not to the figure.

    python bake/bake_dse_front.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bake.bake_common import DATA_DIR, report, require, write_artifact  # noqa: E402
from noc_data import derived_rel, load_true_pareto_front_raw  # noqa: E402

DSE_DIR = DATA_DIR / "dse_12r3c"
PARTS = ("dataset_all_X", "dataset_all_sat", "dataset_all_pow")


def bake() -> Path:
    srcs = [require(DSE_DIR / p, f"dse_12r3c/{p}") for p in PARTS]
    sat, pow_min, area_min = load_true_pareto_front_raw(DSE_DIR)
    out = write_artifact(
        derived_rel(DSE_DIR, "_front.npz"), srcs,
        {"sat": sat, "pow_min": pow_min, "area_min": area_min},
        baked_by="bake/bake_dse_front.py",
    )
    print(f"dse_12r3c: {len(sat)} distinct saturation values on the front")
    report(out, srcs)
    return out


if __name__ == "__main__":
    raise SystemExit(bake() and 0)
