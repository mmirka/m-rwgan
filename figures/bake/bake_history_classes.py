#!/usr/bin/env python3
"""Bake a ``*_historyFake[_fix]`` training log down to per-router class indices.

A ``historyFake`` pickle is the generator's raw output on a fixed noise vector,
logged once per epoch: 301 epochs x 100 samples x 64 routers x n_features
float32. For the uniform Sat-only run that is 516 MB, because its GCN
generator concatenates a 64-wide one-hot node ID onto the 3 class scores.

Every figure that reads one — ``router_evolution.py --type training-progress``
(Fig 6.9), ``router_size_maps.py`` (4uni_t2 / 4hot_t2) and
``taux_routeurs_date2022.py`` (the BDtime* curves) — reaches it through
``noc_data.per_router_class_fraction``, whose first act is
``argmax(sample[router, :n_classes])``. Everything downstream is a count of
those argmaxes. So the sufficient statistic is one small integer per
(epoch, sample, router), and the float features are the input to the argmax,
not to the figure.

Baked shape: ``(n_epochs, batch, n_routers)`` int8. 516 MB becomes ~2 MB
uncompressed and far less on disk, and the argmax is exact — the figures
rebuild bit-identically, which is what ``bake/verify.sh`` checks.

    python bake/bake_history_classes.py            # every historyFake in the sweep tree
    python bake/bake_history_classes.py <path>...  # named files only
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bake.bake_common import DATA_DIR, report, write_artifact  # noqa: E402
from noc_data import as_per_router_samples, derived_rel  # noqa: E402

SWEEP_DIR = DATA_DIR / "reward_weight_sweep_8x8_3reward"

N_ROUTERS = 8 * 8   # every archived run is the 8x8 mesh
N_CLASSES = 3       # Big / Medium / Small, the 3-class setup of the headline results


def class_indices(history, n_routers: int, n_classes: int) -> np.ndarray:
    """``(n_epochs, batch, n_routers)`` int8 of per-router argmax class indices."""
    epochs = []
    batch = None
    for i, entry in enumerate(history):
        samples = as_per_router_samples(entry, n_routers)
        if samples.shape[-1] < n_classes:
            raise ValueError(
                f"epoch {i}: {samples.shape[-1]} feature columns, need at least {n_classes}"
            )
        idx = np.argmax(samples[:, :, :n_classes], axis=2).astype(np.int8)
        if batch is None:
            batch = idx.shape[0]
        elif idx.shape[0] != batch:
            # The readers index history[epoch] and take len(history); a ragged
            # batch axis would not survive being stacked into one array.
            raise ValueError(
                f"epoch {i} has batch {idx.shape[0]}, epoch 0 had {batch}; "
                "this history cannot be baked as a single array"
            )
        epochs.append(idx)
    return np.stack(epochs)


def bake(src: Path) -> Path:
    with open(src, "rb") as f:
        history = pickle.load(f)
    idx = class_indices(history, N_ROUTERS, N_CLASSES)
    out = write_artifact(
        derived_rel(src, "_classes.npz"), src,
        {
            "classes": idx,
            "n_classes": np.int64(N_CLASSES),
            "n_routers": np.int64(N_ROUTERS),
        },
        baked_by="bake/bake_history_classes.py",
    )
    print(f"{src.name}: {idx.shape[0]} epochs x {idx.shape[1]} samples x {idx.shape[2]} routers")
    report(out, src)
    return out


def main(argv=None) -> int:
    args = list(argv or sys.argv[1:])
    if args:
        srcs = [Path(a) for a in args]
    else:
        if not SWEEP_DIR.exists():
            print(
                f"error: {SWEEP_DIR} not found.\n"
                "This tree is git-ignored — see figures/data/README.md for the download.",
                file=sys.stderr,
            )
            return 1
        srcs = sorted(SWEEP_DIR.glob("*/*_historyFake")) + sorted(SWEEP_DIR.glob("*/*_historyFake_fix"))
    if not srcs:
        print(f"error: no *_historyFake files under {SWEEP_DIR}", file=sys.stderr)
        return 1
    for src in srcs:
        bake(src)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
