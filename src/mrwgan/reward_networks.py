"""Reward-network architectures: frozen critics used to steer the M-RWGAN generator.

Three reward networks:
- GCN, saturation/throughput target
- GCN, power target (identical architecture to the saturation net, different
  training target)
- CNN, area target

All three networks are regressors (single sigmoid output, MAE loss) trained
on a fixed 8x8-mesh router-class assignment. The GCN nets additionally take
the mesh adjacency matrix as input; the CNN net treats the router-class grid
as a `(8, 8, F)` image.
"""
from __future__ import annotations

import tensorflow as tf
from tensorflow.keras.layers import (
    Convolution2D,
    Dense,
    Dropout,
    Flatten,
    Input,
    LeakyReLU,
)
from tensorflow.keras.models import Model, Sequential

LEAKY_RELU_ALPHA = 0.2


def build_gcn_reward_net(n_nodes: int, n_features: int, dropout_rate: float = 0.2) -> Model:
    """3-layer GCN + global-attention pooling + sigmoid regression head.

    Used for both the saturation/throughput and power reward networks
    (``gcn_hotspot30_10k_3c_sat__*`` / ``gcn_hotspot30_10k_3c_pow__*`` in
    ``data/reference/reward_networks/checkpoints/``).

    NOTE: this function uses a plain ``GlobalSumPool`` here, not
    ``GlobalAttnSumPool``. The saved checkpoints under ``data/reference/``
    contain a plain ``GlobalSumPool`` layer (verified by inspecting
    ``saved_model.pb`` directly), so this architecture is built to match
    the checkpoints.
    """
    from spektral.layers import GCNConv, GlobalSumPool

    leaky_relu = LeakyReLU(alpha=LEAKY_RELU_ALPHA)

    X_in = Input(shape=(n_nodes, n_features))
    A_in = Input(shape=(n_nodes, n_nodes))

    l2_reg = 2.5e-4
    reg = tf.keras.regularizers.l2(l2_reg)

    dr0 = Dropout(rate=dropout_rate)(X_in)
    x1 = GCNConv(128, activation=leaky_relu)([dr0, A_in])
    dr1 = Dropout(rate=dropout_rate)(x1)
    x2 = GCNConv(64, activation=leaky_relu)([dr1, A_in])
    dr2 = Dropout(rate=dropout_rate)(x2)
    x3 = GCNConv(64, activation=leaky_relu)([dr2, A_in])
    x4 = GCNConv(32, activation=leaky_relu, kernel_regularizer=reg)([dr2, A_in])

    gnn = Model(inputs=[X_in, A_in], outputs=x4, name="gnn")
    gnn_out = gnn([X_in, A_in])
    pooled = GlobalSumPool()(gnn_out)
    output = Dense(1, activation="sigmoid")(pooled)

    return Model(inputs=[X_in, A_in], outputs=output, name="gcn_reward_net")


def build_cnn_reward_net(rows: int, cols: int, n_features: int) -> Model:
    """2-layer CNN + dense head, sigmoid regression output.

    Used for the area reward network (``cnn_hotspot30_10k_3c_area_sigmoid__*``
    in ``data/reference/reward_networks/checkpoints/``). Input is the
    router-class grid reshaped to `(rows, cols, n_features)`.
    """
    model = Sequential(name="cnn_reward_net")
    model.add(Convolution2D(128, (3, 3), padding="same", input_shape=(rows, cols, n_features)))
    model.add(LeakyReLU(alpha=LEAKY_RELU_ALPHA))
    model.add(Convolution2D(32, (3, 3), strides=[2, 2], padding="same"))
    model.add(LeakyReLU(alpha=LEAKY_RELU_ALPHA))
    model.add(Flatten())
    model.add(Dense(128))
    model.add(LeakyReLU(alpha=LEAKY_RELU_ALPHA))
    model.add(Dense(1, activation="sigmoid"))

    img = Input(shape=(rows, cols, n_features))
    score = model(img)
    return Model(img, score, name="cnn_reward_net_wrapped")


# custom_objects required by tf.keras.models.load_model for the checkpoints
# under data/reference/reward_networks/checkpoints/.
def reward_net_custom_objects() -> dict:
    from spektral.layers import GCNConv, GlobalSumPool

    return {"GCNConv": GCNConv, "LeakyReLU": LeakyReLU, "GlobalSumPool": GlobalSumPool}


def load_reward_network(checkpoint_dir: str, name: str | None = None, trainable: bool = False) -> Model:
    """Load one frozen reward network `SavedModel` checkpoint.

    `trainable=False` matches how the reward nets are always used inside
    M-RWGAN training (they are frozen critics).
    """
    model = tf.keras.models.load_model(checkpoint_dir, custom_objects=reward_net_custom_objects())
    model.trainable = trainable
    if name is not None:
        model._name = name
    return model
