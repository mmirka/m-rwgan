"""Shared data types and dataset-loading helpers.

``Data_AX`` must be importable from here (not redefined ad hoc) for any
script in this repo to unpickle the files under ``data/reference/`` and
``data/raw/``. Those pickles were created in contexts where ``Data_AX``
lived in the ``__main__`` module, so a plain ``pickle.load`` on them fails
unless ``Data_AX`` is registered under ``__main__`` (see
``load_data_ax_dataset`` below).
"""
from __future__ import annotations

import numpy as np

# Router-class codes used throughout the project (see
# reference/hnocs_pipeline/routers.py for the full router taxonomy).
# 3-class setup ("Big"/"Medium"/"Small" homogeneous routers), used for the
# paper's headline results:
ROUTER_CLASS_MAP_3 = {112: 0, 104: 1, 102: 2}
# 8-class setup used in earlier exploratory work (kept for completeness,
# not used by any script in this repo):
ROUTER_CLASS_MAP_8 = {112: 0, 102: 1, 42: 2, 43: 3, 44: 4, 45: 5, 46: 6, 47: 7}


class Data_AX:
    """One simulated NoC sample: topology + router classes + measured performance.

    Attributes mirror the HNOCS/Orion3.0 simulation output as assembled by
    ``reference/hnocs_pipeline/save_dataset.py``.
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


def router_counter(X, r_type: int, class_map: dict = ROUTER_CLASS_MAP_3) -> int:
    """Count routers of a given class in a raw router-code assignment `X`."""
    return sum(1 for r in X if class_map[r] == r_type)


def one_hot_router_classes(X, nb_classes: int, class_map: dict = ROUTER_CLASS_MAP_3) -> np.ndarray:
    """Convert raw router codes (e.g. 112/104/102) to one-hot class vectors."""
    out = []
    for code in X:
        vec = [0] * nb_classes
        vec[class_map[code]] = 1
        out.append(vec)
    return np.asarray(out)


class NodeIDTransform:
    """Spektral-style graph transform: concatenate a one-hot node ID to node features.

    Required so the GCN reward nets and the M-RWGAN critic can distinguish
    otherwise-identical router positions in the mesh.
    """

    def __init__(self, max_id: int):
        self.max_id = max_id

    def __call__(self, graph):
        from spektral.utils import one_hot

        if "a" not in graph:
            raise ValueError("The graph must have an adjacency matrix")
        ids = one_hot(range(0, self.max_id + 1), self.max_id + 1)
        if "x" not in graph:
            graph.x = ids
        else:
            graph.x = np.concatenate((graph.x, ids), axis=-1)
        return graph


def load_data_ax_dataset(path: str) -> list:
    """Load a pickled list of `Data_AX` objects (the format used throughout
    `data/reference/` and `data/raw/`).

    These pickles were created in contexts where `Data_AX` lived in the
    `__main__` module, so each one records its class as `__main__.Data_AX`,
    not `mrwgan.data_types.Data_AX` — a plain `pickle.load` fails unless
    `Data_AX` is reachable under `__main__`. Register it there here so
    every caller can unpickle real data regardless of how it's invoked.
    """
    import pickle
    import sys

    sys.modules["__main__"].Data_AX = Data_AX
    with open(path, "rb") as f:
        return pickle.load(f)


def build_gcn_dataset(raw_samples: list, nb_classes: int, max_id: int, targets=None):
    """Build a Spektral `Dataset` of router-class graphs, GCN-normalized.

    Pipeline shared by the critic and the GCN reward nets: one-hot router
    classes -> `GCNFilter` (renormalizes the adjacency the way `GCNConv`
    expects) -> concatenate one-hot node ID via `NodeIDTransform`.

    `targets`, if given, is a 1D array of per-sample regression targets
    (e.g. normalized saturation or power) attached as each graph's `y`; the
    resulting node features (`.x`, shape `(n_nodes, nb_classes + max_id + 1)`)
    and normalized adjacency (`.a`) are what both the reward-network critics
    and the M-RWGAN critic/generator actually consume — NOT the raw
    3-class one-hot alone.
    """
    from spektral.data import Dataset, Graph
    from spektral.transforms import GCNFilter

    class _RouterDataset(Dataset):
        def read(self):
            output = []
            for i, sample in enumerate(raw_samples):
                x = one_hot_router_classes(sample.X, nb_classes)
                a = np.asarray(sample.A, dtype=np.int32)
                y = targets[i] if targets is not None else None
                output.append(Graph(a=a, x=x, y=y))
            return output

    dataset = _RouterDataset()
    dataset.apply(GCNFilter())
    dataset.apply(NodeIDTransform(max_id))
    return dataset
