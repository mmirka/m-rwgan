"""Minimal, numpy-only data helpers for the paper-figure scripts.

Just enough of the M-RWGAN data layer to load the historical ``Data_AX``
pickles under ``figures/data/`` and derive the per-topology summaries the
figures plot. No TensorFlow, no spektral: the figure scripts never touch the
GCN/critic path, only the stored simulation results.

``Data_AX`` must be importable (not redefined ad hoc) to unpickle those
files. They were pickled in contexts where ``Data_AX`` lived in the
top-level ``__main__`` module, so each pickle records its class as
``__main__.Data_AX``; ``load_data_ax_dataset`` registers this class under
``__main__`` before loading so a plain ``pickle.load`` succeeds regardless
of how the caller is invoked.

Four of the inputs are too large for git (~1.2 GB in total). For each of
them ``figures/bake/`` writes a small, git-tracked artifact under
``figures/data/derived/`` holding exactly the reduction the figures consume,
and the loaders here look there first — so a plain clone rebuilds every
figure with no download, while pointing a script's ``--…-path`` override at
the raw archive still works and still takes precedence.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
DERIVED_DIR = DATA_DIR / "derived"

# 3-class router setup ("Big"/"Medium"/"Small" homogeneous routers), used for
# the paper's headline results. Maps raw router codes to class indices.
ROUTER_CLASS_MAP_3 = {112: 0, 104: 1, 102: 2}

# Per-router buffer size by raw router code (Big / Medium / Small) and, in the
# ROUTER_CLASS_MAP_3 index order, by class index.
ROUTER_BUFFER_SIZE = {112: 12, 104: 4, 102: 2}
BUFFER_SIZE_BY_CLASS_INDEX = [12, 4, 2]

_N_ROUTERS = 64
AREA_MIN = _N_ROUTERS * min(ROUTER_BUFFER_SIZE.values())
AREA_MAX = _N_ROUTERS * max(ROUTER_BUFFER_SIZE.values())


class Data_AX:
    """One simulated NoC sample: topology + router classes + measured performance.

    Attributes mirror the HNOCS/Orion3.0 simulation output (see
    ``reference/hnocs_pipeline/`` for how the raw data was produced).
    """

    def __init__(self, A, X, latency, total_power, powers, total_area, areas, total_jouls=None):
        self.A = A                    # adjacency matrix
        self.X = X                    # router-class assignment (raw codes, e.g. 112/104/102)
        self.latency = latency        # [injection_rate, latency] curve
        self.total_power = total_power
        self.powers = powers
        self.total_area = total_area
        self.areas = areas
        self.total_jouls = total_jouls


def saturation_point(latencies: np.ndarray, threshold_multiplier: float = 3.0) -> float:
    """Injection rate at which latency exceeds `threshold_multiplier`x the idle latency."""
    start = latencies[1, 0]
    for i in range(latencies.shape[1]):
        lat = latencies[1, i]
        ir = latencies[0, i]
        if lat > threshold_multiplier * start:
            return ir
    return 0.0


def normalized_area(x_codes) -> float:
    """Buffer-weighted, min-max-normalized router-area proxy in [0, 1].

    Sum of per-router buffer sizes (Big=12, Medium=4, Small=2), normalized
    over the theoretical [all-Small, all-Big] range for a 64-router mesh.

    Handles both `X` representations in this repo's data: raw router codes
    (baseline topologies, e.g. `112`) and per-router soft-class(+node-ID)
    feature vectors (M-RWGAN generator output under `figures/data/generated/`
    — the class one-hot/softmax always occupies the first `n_classes`
    columns).
    """
    first = x_codes[0]
    if np.ndim(first) > 0:  # per-router feature vectors, not raw codes
        n_classes = len(BUFFER_SIZE_BY_CLASS_INDEX)
        raw = sum(BUFFER_SIZE_BY_CLASS_INDEX[np.argmax(row[:n_classes])] for row in x_codes)
    else:
        raw = sum(ROUTER_BUFFER_SIZE[c] for c in x_codes)
    return (raw - AREA_MIN) / (AREA_MAX - AREA_MIN)


def resolve_data_path(explicit, bundle_path, *, what: str) -> Path:
    """The ``explicit`` CLI path if given, else the ``figures/data/`` bundle path.

    ``bundle_path`` is an absolute path under ``figures/data/`` — the only place
    the figure scripts look. Raises ``FileNotFoundError`` naming it (and
    ``figures/data/README.md``) when it does not exist: a few of the bundled
    inputs are too large for git and must be downloaded into ``figures/data/``
    first, or pointed at with the script's own ``--…-path`` override.
    """
    if explicit:
        p = Path(explicit)
        if p.exists():
            return p
        raise FileNotFoundError(f"{what}: path {p} does not exist.")
    bundle_path = Path(bundle_path)
    if bundle_path.exists():
        return bundle_path
    raise FileNotFoundError(
        f"{what}: input not found at {bundle_path}.\n"
        "Some inputs are git-ignored — see figures/data/README.md for the download."
    )


def derived_rel(source_path, suffix: str) -> Path:
    """Where ``bake/`` writes the artifact for ``source_path``, relative to ``derived/``.

    The artifact mirrors its raw input's path under ``figures/data/`` rather
    than just its basename: two runs in the archive share a filename across the
    ``uniform_3c`` and ``hotspot30_3c`` subdirectories, so a flat derived
    directory would have one silently overwrite the other. Mirroring also keeps
    every artifact traceable to the archive it summarises by path alone.
    """
    source_path = Path(source_path)
    try:
        rel = source_path.resolve().relative_to(DATA_DIR.resolve())
    except ValueError:  # a path outside figures/data/ — key it by name alone
        rel = Path(source_path.name)
    return rel.parent / f"{rel.name}{suffix}"


def derived_for(source_path, suffix: str) -> Path:
    """Absolute path of the derived artifact for ``source_path``."""
    return DERIVED_DIR / derived_rel(source_path, suffix)


def _open_derived(path: Path, what: str):
    z = np.load(path, allow_pickle=False)
    version = int(z["_format_version"])
    if version != 1:
        z.close()
        raise ValueError(
            f"{what}: {path.name} is format v{version}, this reader understands v1. "
            "Re-bake it with the matching bake/ script."
        )
    return z


def load_dataset_summary(explicit, bundle_path, *, what: str) -> dict:
    """``{'sat', 'pow', 'area'}`` for a ``dataset_10k_*`` set, derived tier first.

    Resolution order:

    1. ``explicit`` — a raw pickle the caller named on the command line. An
       explicit path is honoured as given: it is how you point a figure at a
       copy of the archive held elsewhere.
    2. ``data/derived/<name>_summary.npz`` — git-tracked, ~250 KB, written by
       ``bake/bake_dataset_summary.py``. This is what a plain clone has.
    3. ``bundle_path`` — the raw 250-320 MB pickle, if it was downloaded.

    Raises ``FileNotFoundError`` naming both tiers when neither is present.
    """
    if explicit:
        return summarize_samples(load_data_ax_dataset(str(resolve_data_path(
            explicit, bundle_path, what=what
        ))))
    bundle_path = Path(bundle_path)
    derived = derived_for(bundle_path, "_summary.npz")
    if derived.exists():
        with _open_derived(derived, what) as z:
            return {m: z[m] for m in ("sat", "pow", "area")}
    if bundle_path.exists():
        return summarize_samples(load_data_ax_dataset(str(bundle_path)))
    raise FileNotFoundError(
        f"{what}: neither the derived summary at {derived} nor the raw dataset at "
        f"{bundle_path} exists.\n"
        "The derived artifact ships with the repository — if it is missing, the clone is "
        "incomplete. To rebuild it, download the raw dataset (figures/data/README.md) and "
        "run bake/bake_dataset_summary.py."
    )


def load_history(path) -> "list | ClassIndices":
    """One run's per-epoch generator batches, from a baked ``.npz`` or a raw pickle.

    Both forms are indexed the same way by the figure scripts: ``len(history)``
    is the epoch count and ``history[epoch]`` is one epoch's batch.
    """
    path = Path(path)
    if path.suffix == ".npz":
        with _open_derived(path, "history") as z:
            out = z["classes"].view(ClassIndices)
            out.n_classes = int(z["n_classes"])
            out.n_routers = int(z["n_routers"])
            return out
    with open(path, "rb") as f:
        return pickle.load(f)


def resolve_history(explicit, bundle_path, *, what: str) -> Path:
    """Path to one run's history: ``explicit``, else the baked artifact, else the raw pickle."""
    if explicit:
        return resolve_data_path(explicit, bundle_path, what=what)
    bundle_path = Path(bundle_path)
    derived = derived_for(bundle_path, "_classes.npz")
    if derived.exists():
        return derived
    if bundle_path.exists():
        return bundle_path
    raise FileNotFoundError(
        f"{what}: neither the baked history at {derived} nor the raw pickle at "
        f"{bundle_path} exists.\n"
        "The baked artifact ships with the repository — if it is missing, the clone is "
        "incomplete. To rebuild it, download the raw history (figures/data/README.md) and "
        "run bake/bake_history_classes.py."
    )


def find_history(base_dir, reward_ratio: str) -> Path | None:
    """Best history for ``reward_ratio`` under ``base_dir``: baked first, then raw.

    ``base_dir`` names the *raw* directory; its baked counterpart is the same
    path under ``data/derived/``. Within each tier a ``_historyFake_fix`` run
    wins over a plain ``_historyFake``, which is the preference the raw-only
    lookup had.
    """
    base_dir = Path(base_dir)
    baked_dir = derived_for(base_dir, "")
    tiers = (
        (baked_dir, f"*{reward_ratio}*_historyFake_fix_classes.npz"),
        (baked_dir, f"*{reward_ratio}*_historyFake_classes.npz"),
        (base_dir, f"*{reward_ratio}*_historyFake_fix"),
        (base_dir, f"*{reward_ratio}*_historyFake"),
    )
    for directory, pattern in tiers:
        if not directory.exists():
            continue
        matches = sorted(directory.glob(pattern))
        if matches:
            return matches[0]
    return None


def load_true_pareto_front_raw(dse_dir: Path):
    """The marginal "true Pareto front" of the 12-router/3-class enumeration.

    Loads the full 531,441-NoC design space, normalizes saturation/power/area
    to [0, 1] over that whole set (thesis: "L'ensemble des resultats ... est
    normalise entre 0 et 1 en fonction des minimum et maximum de chaque
    mesure"), then for each distinct saturation value takes the minimum power
    and minimum area — the reference set the generated NoCs are scored against
    in Fig 6.20.

    This is the reduction ``bake/bake_dse_front.py`` performs once; at figure
    time ``load_true_pareto_front`` reads its result instead.
    """
    dse_dir = Path(dse_dir)

    def _unpickle(name):
        with open(dse_dir / name, "rb") as f:
            return pickle.load(f)

    X = np.asarray(_unpickle("dataset_all_X"))
    sat_raw = np.asarray(_unpickle("dataset_all_sat"), dtype=float)
    pow_raw = np.asarray(_unpickle("dataset_all_pow"), dtype=float)

    y_sat = min_max_normalize(sat_raw, sat_raw.min(), sat_raw.max())
    y_pow = min_max_normalize(pow_raw, pow_raw.min(), pow_raw.max())
    y_area = area_norm_batch(X)

    list_sat, (list_pow_min, list_area_min) = marginal_min_front(y_sat, y_pow, y_area)
    return list_sat, list_pow_min, list_area_min


def load_true_pareto_front(explicit_dse_dir, bundle_dse_dir, *, what: str):
    """The Fig 6.20 reference front, from the baked artifact or the raw enumeration."""
    if explicit_dse_dir:
        return load_true_pareto_front_raw(resolve_data_path(
            explicit_dse_dir, bundle_dse_dir, what=what
        ))
    derived = derived_for(bundle_dse_dir, "_front.npz")
    if derived.exists():
        with _open_derived(derived, what) as z:
            return z["sat"], z["pow_min"], z["area_min"]
    bundle_dse_dir = Path(bundle_dse_dir)
    if bundle_dse_dir.exists():
        return load_true_pareto_front_raw(bundle_dse_dir)
    raise FileNotFoundError(
        f"{what}: neither the baked front at {derived} nor the raw enumeration at "
        f"{bundle_dse_dir} exists.\n"
        "The baked artifact ships with the repository — if it is missing, the clone is "
        "incomplete. To rebuild it, download the enumeration (figures/data/README.md) and "
        "run bake/bake_dse_front.py."
    )


def min_max_normalize(x, vmin, vmax):
    """Notebook ``normalize``: linear rescale so ``vmin`` -> 0, ``vmax`` -> 1 (no clipping)."""
    return (x - vmin) / (vmax - vmin)


class ClassIndices(np.ndarray):
    """Per-router class indices — the baked form of one ``*_historyFake`` batch.

    A plain ndarray view, tagged so ``per_router_class_fraction`` knows the
    ``argmax`` over generator features has already been taken (``bake/
    bake_history_classes.py`` takes it once, at bake time). Indexing a
    ``(n_epochs, batch, n_routers)`` instance yields the ``(batch, n_routers)``
    instance for one epoch, so a baked history slices, iterates and reports
    ``len()`` exactly like the raw pickle's list of per-epoch arrays.

    ``n_classes`` travels with the array because it was fixed at bake time: a
    figure asking for a different class count is asking for a different
    argmax, which the baked data can no longer provide.
    """

    n_classes: int = 0
    n_routers: int = 0

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.n_classes = getattr(obj, "n_classes", 0)
        self.n_routers = getattr(obj, "n_routers", 0)


def as_per_router_samples(entry, n_routers: int, batch_size: int = 100) -> np.ndarray:
    """One logged generator epoch -> ``(batch, n_routers, n_features)``.

    Handles every shape the archived M-RWGAN ``*_historyFake`` /
    ``*_historyFake_fix`` pickles use:

    - ndim 4 ``(batch, mesh_rows, mesh_cols, n_features)`` -> flatten the grid to
      ``(batch, n_routers, n_features)``. ``n_features`` is ``n_classes``, or
      ``n_classes + max_node_id + 1`` for the GCN generator variant that
      concatenates a one-hot node ID (the class one-hot always occupies the
      *first* ``n_classes`` columns).
    - ndim 3 ``(batch, n_routers, n_features)`` -> as-is.
    - ndim 2 ``(batch, n_routers * n_features)`` -> ``(batch, n_routers, -1)``.
    - ndim 1 -> ``(batch_size, n_routers, -1)``.

    The feature axis is never split except in the ndim-4 grid case, where its
    size is known from the input.
    """
    arr = np.asarray(entry)
    if arr.ndim == 4:
        return arr.reshape(arr.shape[0], n_routers, arr.shape[-1])
    if arr.ndim == 3:
        return arr
    if arr.ndim == 2:
        return arr.reshape(arr.shape[0], n_routers, -1)
    if arr.ndim == 1:
        return arr.reshape(batch_size, n_routers, -1)
    raise ValueError(f"Unexpected historyFake sample shape {arr.shape}")


def per_router_class_fraction(entry, n_routers: int, n_classes: int, batch_size: int = 100) -> np.ndarray:
    """``(n_routers, n_classes)``: fraction of the batch assigning each router to each class.

    Per sample, each router's class is ``argmax`` over the first ``n_classes``
    feature columns; counts are divided by the batch size. Equals the notebooks'
    ``txs[:, :, epoch]``.
    """
    if isinstance(entry, ClassIndices):
        return _class_fraction_from_indices(entry, n_routers, n_classes)
    samples = as_per_router_samples(entry, n_routers, batch_size)
    batch = samples.shape[0]
    counts = np.zeros((n_routers, n_classes))
    for sample in samples:
        for router in range(n_routers):
            counts[router, np.argmax(sample[router, :n_classes])] += 1
    return counts / batch


def _class_fraction_from_indices(entry: ClassIndices, n_routers: int, n_classes: int) -> np.ndarray:
    """``per_router_class_fraction`` for one baked epoch — the same counts, pre-argmaxed."""
    if entry.n_classes and entry.n_classes != n_classes:
        raise ValueError(
            f"baked history holds {entry.n_classes}-class indices but {n_classes} were "
            "requested; re-bake with bake/bake_history_classes.py or use the raw history"
        )
    idx = np.asarray(entry)
    if idx.ndim != 2 or idx.shape[1] != n_routers:
        raise ValueError(f"baked epoch has shape {idx.shape}, expected (batch, {n_routers})")
    counts = np.zeros((n_routers, n_classes))
    for row in idx:
        counts[np.arange(n_routers), row] += 1
    return counts / idx.shape[0]


def class_fraction_over_epochs(history, n_routers: int, n_classes: int, batch_size: int = 100) -> np.ndarray:
    """``(n_classes, n_epochs)``: mean over routers of ``per_router_class_fraction`` per epoch.

    Equals the notebooks' ``txs_classe``.
    """
    out = np.zeros((n_classes, len(history)))
    for epoch, entry in enumerate(history):
        out[:, epoch] = per_router_class_fraction(entry, n_routers, n_classes, batch_size).mean(axis=0)
    return out


def class_score_map(fraction_by_router, mesh_rows: int, mesh_cols: int, score=(1.0, 0.2, 0.0)) -> np.ndarray:
    """``(mesh_rows, mesh_cols)`` weighted "router size" score -- the notebooks' ``score_img``.

    ``score`` weights the per-router class fractions (Big=1, Medium=0.2,
    Small=0); the weighted sum per router is reshaped onto the mesh.
    """
    fraction_by_router = np.asarray(fraction_by_router)
    score = np.asarray(score[: fraction_by_router.shape[1]])
    return (fraction_by_router @ score).reshape(mesh_rows, mesh_cols)


def summarize_samples(samples, round_sat: int = 5) -> dict:
    """Per-``Data_AX`` saturation / power / area, as three parallel 1-D arrays.

    - ``sat``  = ``round(saturation_point(asarray(s.latency)), round_sat)``
    - ``pow``  = ``asarray(s.total_power)[-1, -1]``
    - ``area`` = ``normalized_area(s.X)``

    The ``round`` mirrors the notebooks, whose dataset-curve binning groups
    samples by the rounded saturation value.
    """
    return {
        "sat": np.array([round(saturation_point(np.asarray(s.latency)), round_sat) for s in samples]),
        "pow": np.array([np.asarray(s.total_power)[-1, -1] for s in samples]),
        "area": np.array([normalized_area(s.X) for s in samples]),
    }


def area_norm_batch(X, buffer_by_code: dict | None = None) -> np.ndarray:
    """Vectorized ``normalized_area`` (raw-codes case) for a batch of samples.

    ``X`` is ``(n_samples, n_routers)`` raw router-class codes (e.g. the
    12-router/3-class DSE design-space enumeration, one row per NoC).
    Normalizes over the *actual* ``n_routers`` in ``X`` -- not a hardcoded
    64 -- matching the source notebooks' per-call ``nb_r = len(X)``.
    """
    buffer_by_code = buffer_by_code or ROUTER_BUFFER_SIZE
    X = np.asarray(X)
    lookup = np.zeros(max(buffer_by_code) + 1)
    for code, size in buffer_by_code.items():
        lookup[code] = size
    raw = lookup[X].sum(axis=1)
    n_routers = X.shape[1]
    area_min = n_routers * min(buffer_by_code.values())
    area_max = n_routers * max(buffer_by_code.values())
    return (raw - area_min) / (area_max - area_min)


def marginal_min_front(bucket: np.ndarray, *values: np.ndarray):
    """Sorted unique ``bucket`` values, paired with the per-bucket min of each ``values`` array.

    Mirrors the notebooks' ``list_sat`` / ``list_satpow_min`` / ``list_satare_min``
    pattern: group all design-space points by (exact) saturation value, and for
    each group take the minimum power / minimum area -- the per-axis marginal
    "true Pareto front" used as the IGD reference set.

    Returns ``(unique_bucket, [min_values_for_each_values_array])``.
    """
    bucket = np.asarray(bucket)
    unique = np.sort(np.unique(bucket))
    mins = [np.array([np.asarray(v)[bucket == u].min() for u in unique]) for v in values]
    return unique, mins


def euclidean_igd_best(samples: np.ndarray, ref_x: np.ndarray, ref_y: np.ndarray, ref_z: np.ndarray):
    """Best (min-distance) 3D euclidean match of ``samples`` against a reference front.

    ``samples`` is ``(n_samples, 3)`` normalized ``[x, y, z]`` points (e.g. one
    training run's 100 generated NoCs); ``ref_x``/``ref_y``/``ref_z`` are equal-length
    1-D arrays describing the reference front. Returns ``(best_distance,
    per_axis_abs_deltas)`` for the single (sample, reference-point) pair with the
    smallest euclidean distance -- ties broken by row-major order, matching the
    source notebooks' nested-loop search (samples outer, reference points inner).
    """
    ref = np.stack([np.asarray(ref_x), np.asarray(ref_y), np.asarray(ref_z)], axis=1)  # (n_ref, 3)
    diffs = np.asarray(samples)[:, None, :] - ref[None, :, :]  # (n_samples, n_ref, 3)
    dists = np.sqrt((diffs**2).sum(axis=2))
    idx = np.unravel_index(np.argmin(dists), dists.shape)
    return float(dists[idx]), np.abs(diffs[idx])


def load_data_ax_dataset(path: str) -> list:
    """Load a pickled list of `Data_AX` objects (the format used under `figures/data/`).

    Registers `Data_AX` under `__main__` first so pickles that recorded their
    class as `__main__.Data_AX` unpickle regardless of how this is invoked.
    """
    import pickle
    import sys

    sys.modules["__main__"].Data_AX = Data_AX
    with open(path, "rb") as f:
        return pickle.load(f)
