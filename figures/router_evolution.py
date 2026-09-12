#!/usr/bin/env python3
"""Reproduce the router-class / router-size evolution figures.

DATE2022 / PhD thesis Chapter 6. Two independent figure *types*, selected
with ``--type``:

``--type training-progress``
    Fig 6.9: 5-panel spatial snapshot ("Mean Generated NoC" -- an 8x8
    heatmap of the weighted router-class score, i.e. "router's size") at
    training epochs 0 / 50 / 100 / 200 / (last), for ONE fixed reward-weight
    run. Selected with ``--config`` (traffic) and ``--reward-ratio`` (which
    run). Reads one run's per-epoch fixed-noise generator sample batches:
    the bundled ``data/derived/*_classes.npz``, or the raw
    ``*_historyFake[_fix]`` pickle under
    ``data/reward_weight_sweep_8x8_3reward/{uniform,hotspot30}_3c/``.

    Fig 6.9 itself = ``--config uniform --type training-progress
    --reward-ratio Sat100``. Note: the thesis text (page 136) confirms Fig
    6.9 is this 5-epoch spatial-snapshot grid, *not* the router-class-
    proportion-over-epoch line plot -- that line plot is actually Fig 6.10c,
    a distinct figure this script does not reproduce (it is not in the
    target list). A prior version of this script's docstring conflated the
    two; this rewrite corrects that.

``--type reward-ratio-compare``
    Figs 6.11 / 6.12 / 6.14 / 6.15 / 6.16 / 6.17: a 7-panel (4+3) grid of the
    same "router's size" heatmap, one panel per point of a Sat-vs-other
    reward-weight sweep (100-0, 90-10, 70-30, 50-50, 30-70, 10-90, 0-100),
    for one traffic (``--config``) and one reward pair (``--reward-pair``,
    default ``both`` -> writes two PNGs, one per pair). Reads the fully
    *simulated* ``Data_AX`` dataset for each sweep point under
    ``data/generated/10-50/<traffic>_<pair>_<model>_<ratio>_dataset`` (NOT
    historyFake -- see "Data source" below) and averages each router's
    class assignment across the whole dataset (the notebook's ``Xs`` /
    ``txs`` / ``hot_img`` pipeline, filtering the full generated dataset by
    reward-weight run rather than by a training epoch).

Fig -> (--config, --type, --reward-ratio / --reward-pair) mapping::

    Fig 6.9  -> --config uniform    --type training-progress      --reward-ratio Sat100
    Fig 6.11 -> --config uniform    --type reward-ratio-compare   --reward-pair SatAndPow
    Fig 6.12 -> --config uniform    --type reward-ratio-compare   --reward-pair SatAndArea
    Fig 6.14 -> --config hotspot    --type reward-ratio-compare   --reward-pair SatAndPow
    Fig 6.15 -> --config hotspot    --type reward-ratio-compare   --reward-pair SatAndArea
    Fig 6.16 -> --config hotspot-cnn --type reward-ratio-compare  --reward-pair SatAndPow
    Fig 6.17 -> --config hotspot-cnn --type reward-ratio-compare  --reward-pair SatAndArea

``--config`` selects the traffic pattern *and*, for reward-ratio-compare,
which reward network the sweep was trained with: ``uniform``/``hotspot``
sweeps use a GCN-based Sat/Pow reward (Area reward is always a CNN);
``hotspot-cnn`` is the same hotspot30 traffic but with a CNN-based Sat/Pow
reward too ("Rewards reposant sur un CNN", thesis p.127) -- there is no
"uniform-cnn" variant, matching the bundled data and the thesis (which only
runs the CNN-reward ablation on hotspot30).

Data source for reward-ratio-compare (verified against the notebook,
``picture_Dataset.ipynb``): cells 15-21 build ``Xs`` by filtering the
*generated dataset* (a plain list of ``Data_AX``, i.e. already-simulated
generator output -- not a historyFake training-history pickle), then derive
``txs``/``txs_classe`` (per-router class fractions, same as this module's
``noc_data.per_router_class_fraction``) and ``hot_img``/``score_img`` (the
weighted score reshaped onto the 8x8 mesh, same as ``noc_data.class_score_map``)
from that filtered set. The bundled ``reward_weight_sweep_8x8_3reward/``
directory only has 1-3 individual historyFake runs per traffic (not a full
7-point sweep), while ``data/generated/10-50/<traffic>_<pair>_<model>_<ratio>_dataset``
DOES have the full 7 (or 6) point sweep needed for Figs 6.11/6.12/6.14-6.17 --
confirming the notebook's dataset-filtering approach, not historyFake, is
the actual source for these multi-panel figures.

Missing "100-0" (pure Sat) panel: the bundled uniform/hotspot ``SatAndArea``
sweeps are missing their ``100-0`` point, but thesis tables 6.5/6.6 (and
6.7/6.8) show *identical* Sat100 numbers for the SatAndPow and SatAndArea
sweeps of a given traffic (a 100% Sat / 0% other-reward run does not depend
on which "other" reward it's nominally paired with). So this script falls
back to the sibling SatAndPow sweep's ``100-0`` dataset for that panel when
the SatAndArea one isn't bundled -- see ``resolve_ratio_dataset``.

Examples::

    python router_evolution.py --config uniform --type training-progress \\
        --reward-ratio Sat100

    python router_evolution.py --config hotspot-cnn --type reward-ratio-compare \\
        --reward-pair SatAndPow

Area/score weighting: "router's size" = weighted sum of per-router
class fractions (Big=1, Medium=0.2, Small=0), matching the thesis colorbar
label and ``noc_data.class_score_map``.
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from noc_data import (  # noqa: E402
    class_score_map,
    find_history,
    load_data_ax_dataset,
    load_history,
    per_router_class_fraction,
)

HERE = Path(__file__).resolve().parent
DATA_ROOT = HERE / "data"
DERIVED_ROOT = DATA_ROOT / "derived"

CONFIGS = ("uniform", "hotspot", "hotspot-cnn")
TYPES = ("training-progress", "reward-ratio-compare")
REWARD_PAIRS = ("SatAndPow", "SatAndArea")

# config -> traffic name used in data/generated/10-50/<traffic>_... filenames
TRAFFIC_BY_CONFIG = {"uniform": "uniform", "hotspot": "hotspot30", "hotspot-cnn": "hotspot30"}

# config -> subdirectory of data/reward_weight_sweep_8x8_3reward/ (training-progress
# only). That whole tree is git-ignored (>500 MB); the per-router class indices
# these figures reduce it to are baked into the tracked data/derived/.
HISTORY_DIR_BY_CONFIG = {"uniform": "uniform_3c", "hotspot": "hotspot30_3c"}

# (config, reward-pair) -> generator/critic model token in data/generated/10-50/ filenames
GENERATED_MODEL = {
    ("uniform", "SatAndPow"): "GCN-GCN",
    ("uniform", "SatAndArea"): "GCN-CNN",
    ("hotspot", "SatAndPow"): "GCN-GCN",
    ("hotspot", "SatAndArea"): "GCN-CNN",
    ("hotspot-cnn", "SatAndPow"): "CNN-CNN",
    ("hotspot-cnn", "SatAndArea"): "CNN-CNN",
}

# panel order matches the thesis: Sat100 first, descending Sat weight to 100% of the other reward
RATIO_ORDER = ("100-0", "90-10", "70-30", "50-50", "30-70", "10-90", "0-100")

# friendly aliases for --reward-ratio (training-progress); value = filename ratio substring
REWARD_RATIO_ALIASES = {"Sat100": "SatAndArea_100-0"}

CLASS_LABELS = ("Big", "Medium", "Small")
TRAINING_EPOCHS = (0, 50, 100, 200, -1)  # matches Fig 6.9's (a)-(e) panels


def ratio_label(ratio: str, reward_b: str) -> str:
    """"100-0" -> "Sat100", "0-100" -> "<reward_b>100", else "Sat<x><reward_b><y>"."""
    sat_pct, other_pct = ratio.split("-")
    if sat_pct == "100":
        return "Sat100"
    if other_pct == "100":
        return f"{reward_b}100"
    return f"Sat{sat_pct}{reward_b}{other_pct}"


def resolve_history_path(config: str, reward_ratio: str) -> Path:
    """Find the history artifact (baked) or pickle (raw) for --config/--reward-ratio.

    ``reward_ratio`` is matched as a substring of the ratio segment of the
    bundled filenames (e.g. ``SatAndArea_100-0``, ``SatAndPowAndArea_50-10-40``),
    after resolving ``REWARD_RATIO_ALIASES``. Prefers a ``_historyFake_fix``
    file; falls back to plain ``_historyFake`` if no ``_fix`` variant exists
    (both are handled equally by ``noc_data.as_per_router_samples``).

    Every raw ``*_historyFake[_fix]`` file lives under the git-ignored
    ``data/reward_weight_sweep_8x8_3reward/`` tree -- Fig 6.9 itself
    (``--config uniform --reward-ratio Sat100``) reads the ~493 MB
    ``uniform_3c/…SatAndArea_100-0…_historyFake``. ``noc_data.find_history``
    looks in the tracked ``data/derived/*_classes.npz`` first, so the default
    lookup needs no download; pass ``--history-path`` to read a raw pickle.
    """
    reward_ratio = REWARD_RATIO_ALIASES.get(reward_ratio, reward_ratio)
    subdir = HISTORY_DIR_BY_CONFIG.get(config)
    if subdir is None:
        raise FileNotFoundError(
            f"training-progress: no reward_weight_sweep_8x8_3reward data bundled for "
            f"--config {config} (only 'uniform' and 'hotspot' have historyFake runs; "
            "'hotspot-cnn' training-progress data was not archived)."
        )
    base_dir = DATA_ROOT / "reward_weight_sweep_8x8_3reward" / subdir
    found = find_history(base_dir, reward_ratio)
    if found:
        return found
    baked_dir = DERIVED_ROOT / base_dir.relative_to(DATA_ROOT)
    available = sorted(
        {p.name.split("_date")[0] for p in baked_dir.glob("*_historyFake*_classes.npz")}
        | {p.name.split("_date")[0] for p in base_dir.glob("*_historyFake*")}
    )
    raise FileNotFoundError(
        f"training-progress: no baked or raw history matching ratio "
        f"'{reward_ratio}' under {baked_dir} or {base_dir}. Runs present:\n  "
        + "\n  ".join(available) +
        "\n(The baked artifacts ship with the repository; the raw tree is git-ignored "
        "-- >500MB, see data/README.md for the download.)"
    )


def resolve_ratio_dataset(traffic: str, config: str, pair: str, model: str, ratio: str) -> Path | None:
    """Find the generated ``Data_AX`` dataset for one reward-ratio-compare panel.

    Falls back to the sibling SatAndPow sweep's ``100-0`` dataset when the
    requested pair's own ``100-0`` point isn't bundled -- see the module
    docstring ("Missing 100-0 panel"). Returns ``None`` (caller blanks the
    panel) if nothing usable is found.
    """
    base_dir = DATA_ROOT / "generated" / "10-50"
    path = base_dir / f"{traffic}_{pair}_{model}_{ratio}_dataset"
    if path.exists():
        return path
    if ratio == "100-0" and pair == "SatAndArea":
        fallback_model = GENERATED_MODEL[(config, "SatAndPow")]
        fallback = base_dir / f"{traffic}_SatAndPow_{fallback_model}_100-0_dataset"
        if fallback.exists():
            print(
                f"  note: {path.name} not bundled; reusing {fallback.name} for the "
                "Sat100 panel (a 100%-Sat/0%-other run doesn't depend on the pairing, "
                "see tables 6.5/6.6 in the thesis)."
            )
            return fallback
    return None


def dataset_class_score(dataset_path: Path, mesh_rows: int, mesh_cols: int, n_classes: int) -> np.ndarray:
    """Load a Data_AX dataset and reduce it to an (mesh_rows, mesh_cols) router-size score map."""
    samples = load_data_ax_dataset(str(dataset_path))
    n_routers = mesh_rows * mesh_cols
    x_stack = np.stack([np.asarray(s.X) for s in samples])  # (n_samples, n_routers, n_features)
    fraction_by_router = per_router_class_fraction(x_stack, n_routers, n_classes, batch_size=len(samples))
    return class_score_map(fraction_by_router, mesh_rows, mesh_cols)


def plot_training_progress(
    history: list, mesh_rows: int, mesh_cols: int, n_classes: int, output_dir: Path, config: str, reward_ratio: str
) -> Path:
    """Fig 6.9-style figure: 5 spatial snapshots of router size over training epochs."""
    n_routers = mesh_rows * mesh_cols
    n_epochs = len(history)
    epochs = [(e if e >= 0 else n_epochs - 1) for e in TRAINING_EPOCHS if (e if e >= 0 else n_epochs - 1) < n_epochs]

    fig, axes = plt.subplots(1, len(epochs), figsize=(3.2 * len(epochs), 3.2))
    if len(epochs) == 1:
        axes = [axes]
    im = None
    for ax, epoch in zip(axes, epochs):
        fraction_by_router = per_router_class_fraction(history[epoch], n_routers, n_classes)
        grid = class_score_map(fraction_by_router, mesh_rows, mesh_cols)
        im = ax.imshow(grid, vmin=0, vmax=1, cmap="viridis")
        ax.set_title(f"epoch {epoch}")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(im, ax=axes, label="router's size", shrink=0.85)
    fig.suptitle(f"Mean generated NoC over training ({config}, {reward_ratio})")
    out_path = output_dir / f"training_progress_{config}_{reward_ratio}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_reward_ratio_compare(
    config: str, pair: str, mesh_rows: int, mesh_cols: int, n_classes: int, output_dir: Path
) -> Path:
    """Figs 6.11/6.12/6.14/6.15/6.16/6.17-style figure: 7-panel (4+3) reward-sweep grid."""
    traffic = TRAFFIC_BY_CONFIG[config]
    model = GENERATED_MODEL[(config, pair)]
    reward_b = pair.replace("SatAnd", "")  # "Pow" or "Area"

    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    axes_flat = axes.flatten()
    im = None
    n_found = 0
    for ax, ratio in zip(axes_flat, RATIO_ORDER):
        label = ratio_label(ratio, reward_b)
        dataset_path = resolve_ratio_dataset(traffic, config, pair, model, ratio)
        if dataset_path is None:
            ax.axis("off")
            ax.set_title(f"{label}\n(data not bundled)", fontsize=9)
            continue
        grid = dataset_class_score(dataset_path, mesh_rows, mesh_cols, n_classes)
        im = ax.imshow(grid, vmin=0, vmax=1, cmap="viridis")
        ax.set_title(label)
        ax.set_xticks([])
        ax.set_yticks([])
        n_found += 1
    axes_flat[-1].axis("off")  # thesis layout is 4 + 3 panels; 8th grid cell is always empty

    if n_found == 0:
        raise FileNotFoundError(
            f"reward-ratio-compare: no generated datasets found for --config {config} "
            f"--reward-pair {pair} under {DATA_ROOT / 'generated' / '10-50'} "
            f"(expected files like '{traffic}_{pair}_{model}_<ratio>_dataset')."
        )
    fig.colorbar(im, ax=list(axes_flat), label="router's size", shrink=0.7)
    fig.suptitle(f"Router size vs Sat/{reward_b} reward weight ({config}, {model})")
    out_path = output_dir / f"reward_ratio_compare_{config}_{pair}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", required=True, choices=CONFIGS, help="Traffic (+ reward-network) setup.")
    parser.add_argument("--type", required=True, choices=TYPES, dest="fig_type")
    parser.add_argument(
        "--reward-ratio",
        default=None,
        help="Required for --type training-progress (unless --history-path is given). "
        "Which historyFake run to plot: 'Sat100' (alias for the Fig 6.9 uniform/hotspot "
        "Sat-only run), or the ratio segment of a bundled filename, e.g. "
        "'SatAndPowAndArea_50-10-40'. See data/reward_weight_sweep_8x8_3reward/<config>_3c/ "
        "for what's bundled.",
    )
    parser.add_argument(
        "--history-path",
        default=None,
        help="For --type training-progress: an explicit *_historyFake[_fix], "
        "*_classes.npz or history_fake_fix.pkl path, e.g. from your own "
        "scripts/train_mrwgan.py run "
        "(results/logs/<run-name>/history_fake_fix.pkl). Overrides --config/--reward-ratio "
        "based resolution -- --config is still required (used only for the output filename).",
    )
    parser.add_argument(
        "--reward-pair",
        choices=(*REWARD_PAIRS, "both"),
        default="both",
        help="For --type reward-ratio-compare: which Sat-vs-X sweep to plot. "
        "'both' (default) writes one PNG per pair.",
    )
    parser.add_argument("--mesh-rows", type=int, default=8)
    parser.add_argument("--mesh-cols", type=int, default=8)
    parser.add_argument("--nb-classes", type=int, default=3)
    parser.add_argument("--output-dir", default=str(HERE / "output"))
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.fig_type == "training-progress":
        if args.history_path:
            history_path = Path(args.history_path)
            if not history_path.exists():
                raise FileNotFoundError(f"--history-path {history_path} does not exist.")
            reward_ratio_label = args.reward_ratio or "custom"
        elif not args.reward_ratio:
            raise SystemExit(
                "--type training-progress requires --reward-ratio (e.g. --reward-ratio Sat100) "
                "or --history-path. See the module docstring / --help for the Fig 6.9 mapping."
            )
        else:
            history_path = resolve_history_path(args.config, args.reward_ratio)
            reward_ratio_label = args.reward_ratio
        history = load_history(history_path)
        print(f"loaded {len(history)} epochs from {history_path}")
        out_path = plot_training_progress(
            history, args.mesh_rows, args.mesh_cols, args.nb_classes, output_dir, args.config, reward_ratio_label
        )
        print(f"wrote {out_path}")
    else:
        pairs = REWARD_PAIRS if args.reward_pair == "both" else (args.reward_pair,)
        for pair in pairs:
            try:
                out_path = plot_reward_ratio_compare(
                    args.config, pair, args.mesh_rows, args.mesh_cols, args.nb_classes, output_dir
                )
            except FileNotFoundError as exc:
                if args.reward_pair != "both":
                    raise
                print(f"skip --reward-pair {pair}: {exc}")
                continue
            print(f"wrote {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
