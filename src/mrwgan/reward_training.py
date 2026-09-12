"""Training loops for the three reward networks.

All three share one data-loading/normalization pattern (`load_targets`) and
diverge only in (a) which target column they regress and (b) GCN vs CNN
architecture + graph vs image input framing.

Training data requirement
--------------------------
Provide a pickled list of `Data_AX` objects for your 8x8 mesh (see
`data/raw/README.md` for the datasets used in the paper).
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn import preprocessing
from sklearn.utils import shuffle
from tensorflow.keras.losses import MeanAbsoluteError
from tensorflow.keras.optimizers import Adam

from .data_types import Data_AX, build_gcn_dataset, load_data_ax_dataset, saturation_point  # noqa: F401
from .reward_networks import build_cnn_reward_net, build_gcn_reward_net

TARGETS = ("saturation", "power", "area")


@dataclass
class RewardTrainConfig:
    dataset_path: str
    target: str  # one of TARGETS
    output_dir: str
    mesh_rows: int = 8
    mesh_cols: int = 8
    nb_classes: int = 3
    batch_size: int = 16
    epochs: int = 100
    learning_rate: float = 0.001
    dropout_rate: float = 0.2
    train_fraction: float = 0.8
    seed: int | None = None


def _raw_target(sample, target: str) -> float:
    if target == "saturation":
        return saturation_point(np.asarray(sample.latency))
    if target == "power":
        return np.asarray(sample.total_power)[-1, -1]
    if target == "area":
        # Sum of raw router codes (e.g. 112/104/102) — a proxy correlated
        # with physical area since larger router classes carry higher codes.
        return sum(sample.X)
    raise ValueError(f"Unknown target {target!r}, expected one of {TARGETS}")


def load_targets(raw_samples: list, target: str) -> np.ndarray:
    """Extract and min-max-normalize the regression target.

    Saturation is normalized to [0, 1] directly. Power and area are
    normalized then flipped (`1 - x`) so that higher normalized value =
    better (lower power / smaller area); saturation keeps its sign.
    """
    raw = np.asarray([_raw_target(s, target) for s in raw_samples], dtype=np.float64).reshape(-1, 1)
    scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    normed = scaler.fit_transform(raw).reshape(-1)
    if target in ("power", "area"):
        normed = -1 * normed + 1
    return normed.astype(np.float32)


def train_gcn_reward_net(config: RewardTrainConfig) -> dict:
    """Train the saturation or power GCN reward net (see `build_gcn_reward_net`)."""
    from spektral.data import BatchLoader

    raw = load_data_ax_dataset(config.dataset_path)

    targets = load_targets(raw, config.target)
    n_routers = config.mesh_rows * config.mesh_cols
    dataset = build_gcn_dataset(raw, config.nb_classes, max_id=n_routers - 1, targets=targets)

    if config.seed is not None:
        np.random.seed(config.seed)
    np.random.shuffle(dataset)
    split = int(config.train_fraction * len(dataset))
    data_tr, data_te = dataset[:split], dataset[split:]

    loader_tr = BatchLoader(data_tr, batch_size=config.batch_size, epochs=config.epochs)
    loader_te = BatchLoader(data_te, batch_size=config.batch_size)

    model = build_gcn_reward_net(n_routers, dataset.n_node_features, config.dropout_rate)
    optimizer = Adam(learning_rate=config.learning_rate)
    loss_fn = MeanAbsoluteError()

    @tf.function
    def train_on_batch(inputs, target):
        with tf.GradientTape() as tape:
            predictions = model(inputs, training=True)
            loss = loss_fn(target, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss

    def evaluate():
        # loader_te has no `epochs` bound (spektral's BatchLoader then
        # iterates forever), so we must stop manually after one epoch's
        # worth of batches rather than let the `for` loop run itself out.
        losses = []
        for step, (inputs, target) in enumerate(loader_te, start=1):
            predictions = model(inputs, training=False)
            losses.append(np.mean([loss_fn(target[i], predictions[i, :]) for i in range(len(target))]))
            if step == loader_te.steps_per_epoch:
                break
        return float(np.mean(losses))

    history = []
    step = 0
    epoch = 0
    batch_losses = []
    for inputs, target in loader_tr:
        step += 1
        batch_losses.append(train_on_batch(inputs, target))
        if step == loader_tr.steps_per_epoch:
            step = 0
            epoch += 1
            test_loss = evaluate()
            train_loss = float(np.mean(batch_losses))
            print(f"epoch {epoch}/{config.epochs}  train_MAE={train_loss:.4f}  test_MAE={test_loss:.4f}")
            history.append((train_loss, test_loss))
            batch_losses = []

    final_test_loss = evaluate()
    return _save_reward_net(config, model, history, final_test_loss)


def train_cnn_reward_net(config: RewardTrainConfig) -> dict:
    """Train the area CNN reward net (see `build_cnn_reward_net`).

    Unlike the GCN nets, the area net does NOT use the one-hot node-ID
    features (node ID is useless for an area estimate) — it only sees the
    `(rows, cols, nb_classes)` router-class grid.
    """
    raw = load_data_ax_dataset(config.dataset_path)

    targets = load_targets(raw, config.target)
    from .data_types import one_hot_router_classes

    grids = np.asarray(
        [
            one_hot_router_classes(s.X, config.nb_classes).reshape(
                config.mesh_rows, config.mesh_cols, config.nb_classes
            )
            for s in raw
        ],
        dtype=np.float32,
    )

    if config.seed is not None:
        np.random.seed(config.seed)
    grids, targets = shuffle(grids, targets, random_state=config.seed)
    split = int(config.train_fraction * len(grids))
    data_tr, data_te = grids[:split], grids[split:]
    y_tr, y_te = targets[:split], targets[split:]

    model = build_cnn_reward_net(config.mesh_rows, config.mesh_cols, config.nb_classes)
    optimizer = Adam(learning_rate=config.learning_rate)
    loss_fn = MeanAbsoluteError()

    @tf.function
    def train_on_batch(inputs, target):
        with tf.GradientTape() as tape:
            predictions = model(inputs, training=True)
            loss = loss_fn(target, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss

    def evaluate():
        # One silent forward pass over the whole in-memory test set. The area
        # CNN has no Dropout/BatchNorm (see build_cnn_reward_net), so this is
        # numerically identical to the old per-batch model.predict() loop, but
        # without its ~n_batches Keras progress bars per epoch and per-call
        # overhead. Matches how train_gcn_reward_net's evaluate() calls the model.
        # reshape(-1): the net outputs (N, 1); MeanAbsoluteError broadcasts a
        # (N,) target against (N, 1) into an (N, N) matrix, so flatten to (N,).
        predictions = model(data_te, training=False).numpy().reshape(-1)
        return float(loss_fn(y_te, predictions))

    history = []
    n_batches = len(data_tr) // config.batch_size
    for epoch in range(config.epochs):
        data_tr, y_tr = shuffle(data_tr, y_tr, random_state=config.seed)
        batch_losses = []
        for b in range(n_batches):
            s, e = b * config.batch_size, (b + 1) * config.batch_size
            batch_losses.append(train_on_batch(data_tr[s:e], y_tr[s:e]))
        test_loss = evaluate()
        train_loss = float(np.mean(batch_losses))
        print(f"epoch {epoch + 1}/{config.epochs}  train_MAE={train_loss:.4f}  test_MAE={test_loss:.4f}")
        history.append((train_loss, test_loss))

    final_test_loss = evaluate()
    return _save_reward_net(config, model, history, final_test_loss)


def _save_reward_net(config: RewardTrainConfig, model, history, final_test_loss: float) -> dict:
    run_dir = Path(config.output_dir) / "checkpoints" / f"reward_{config.target}"
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / f"reward_{config.target}__MAE{final_test_loss:.4g}"
    model.save(checkpoint_path)

    log_dir = Path(config.output_dir) / "logs" / f"reward_{config.target}"
    log_dir.mkdir(parents=True, exist_ok=True)
    history_path = log_dir / "train_test_loss_history.pkl"
    with open(history_path, "wb") as f:
        pickle.dump(np.asarray(history), f)

    return {
        "checkpoint": str(checkpoint_path),
        "history": str(history_path),
        "final_test_mae": final_test_loss,
    }
