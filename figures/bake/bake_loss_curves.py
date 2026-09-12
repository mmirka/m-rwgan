#!/usr/bin/env python3
"""Bake one training run's loss logs down to the series Fig 6.10 draws.

``loss_curves.py`` reads four to six files per run. Two of them are large:
``*_generator_loss`` and ``*_discriminator_loss`` are ``(n_batches, 4)``
float64 arrays logged once per training batch — 6 to 8 MB each, inside the
git-ignored ``reward_weight_sweep_8x8_3reward/`` tree. Of their four columns
the figure plots exactly one each: ``generator_loss[:, 0]`` (G loss) and
``discriminator_loss[:, 2]`` (critic loss). The ``reward{1,2,3}_loss`` arrays
are one value per epoch and already tiny; they are copied in whole so the
artifact is the complete input to the figure.

This is an addition to PLAN.md §6's table, which listed the datasets, the DSE
enumeration and the histories but not these. Without it Fig 6.10 would be the
one figure a plain clone still could not rebuild, which is the property the
whole phase exists to establish.

Columns are kept at float64, so the figure rebuilds bit-identically.

    python bake/bake_loss_curves.py           # the Fig 6.10 run
    python bake/bake_loss_curves.py <prefix>  # any other run prefix
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bake.bake_common import DATA_DIR, report, write_artifact  # noqa: E402
from noc_data import derived_rel  # noqa: E402

# The run Fig 6.10 itself plots (thesis Ch. 6): uniform traffic, Sat-only
# reward, 2-reward SatAndArea logging.
DEFAULT_PREFIX = (
    DATA_DIR / "reward_weight_sweep_8x8_3reward" / "uniform_3c"
    / "multiRWGAN_R-SatGCN_300e_L0.2-d0.05_beta10_SatAndArea_100-0_date08072021_"
)

# Which column of each (n_batches, 4) array loss_curves.py plots.
G_COLUMN = 0
D_COLUMN = 2


def _load(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def bake(prefix: Path) -> Path:
    g_path = Path(f"{prefix}generator_loss")
    d_path = Path(f"{prefix}discriminator_loss")
    for p in (g_path, d_path):
        if not p.exists():
            print(
                f"error: {p} not found.\n"
                "This tree is git-ignored — see figures/data/README.md for the download.",
                file=sys.stderr,
            )
            raise SystemExit(1)

    srcs = [g_path, d_path]
    arrays = {
        "g_loss": np.asarray(_load(g_path))[:, G_COLUMN].astype("float64"),
        "d_loss": np.asarray(_load(d_path))[:, D_COLUMN].astype("float64"),
        "g_column": np.int64(G_COLUMN),
        "d_column": np.int64(D_COLUMN),
    }
    present = []
    for i in (1, 2, 3):
        r_path = Path(f"{prefix}reward{i}_loss")
        if r_path.exists():
            arrays[f"reward{i}"] = np.asarray(_load(r_path)).astype("float64")
            srcs.append(r_path)
            present.append(i)
    arrays["rewards_present"] = np.array(present, dtype=np.int64)

    out = write_artifact(
        derived_rel(prefix.parent / prefix.name.rstrip("_"), "_losses.npz"),
        srcs, arrays, baked_by="bake/bake_loss_curves.py",
    )
    print(
        f"{prefix.name}: {len(arrays['g_loss'])} batches, "
        f"reward channels {present or 'none'}"
    )
    report(out, srcs)
    return out


def main(argv=None) -> int:
    args = list(argv or sys.argv[1:])
    prefixes = [Path(a) for a in args] or [DEFAULT_PREFIX]
    for prefix in prefixes:
        bake(prefix)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
