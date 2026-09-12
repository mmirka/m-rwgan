"""M-RWGAN generator, critic, and the combined reward-weighted GAN model.

The model-construction code lives here; the training loop itself lives in
``mrwgan.training``.

The generator produces a per-router class assignment (Gumbel-softmax over 3
classes: Big/Medium/Small) for a **fixed** 8x8-mesh adjacency matrix — it
does not generate graph structure, only the per-node labeling.
"""
from __future__ import annotations

import tensorflow as tf
from tensorflow.keras.activations import softmax
from tensorflow.keras.layers import (
    BatchNormalization,
    Concatenate,
    Dense,
    Dropout,
    Input,
    Lambda,
    LeakyReLU,
    Reshape,
)
from tensorflow.keras.models import Model, Sequential

LEAKY_RELU_ALPHA = 0.2
GUMBEL_TEMPERATURE = 1.0


def build_generator(latent_dim: int, size_out: int) -> Model:
    """MLP generator: noise vector -> raw (pre-Gumbel-softmax) router-class logits.

    `size_out` = `n_routers * n_classes` (192 for the paper's 8x8x3 config).
    """
    model = Sequential(name="generator_mlp")
    model.add(Dense(size_out * 4, input_dim=latent_dim))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=LEAKY_RELU_ALPHA))

    model.add(Dense(size_out * 8))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=LEAKY_RELU_ALPHA))

    model.add(Dense(size_out))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=LEAKY_RELU_ALPHA))

    noise = Input(shape=(latent_dim,))
    img = model(noise)
    return Model(noise, img, name="gene_raw")


def make_gumbel_softmax(n_routers: int, n_classes: int, tau: float = GUMBEL_TEMPERATURE):
    """Gumbel-softmax sampler over per-router class logits.

    Returns a function suitable for use inside a Keras `Lambda` layer.
    `tau` (temperature) is a plain Python float and constant: it is never
    annealed during training. A `tf.Variable` closed over inside a `Lambda`
    layer would also break `.h5` checkpoint saving (the layer config isn't
    JSON-serializable) — a plain float avoids that.
    """

    def gumbel_softmax(logits_y):
        u = tf.random.uniform(tf.shape(logits_y), 0, 1)
        y = logits_y - tf.math.log(-tf.math.log(u + 1e-20) + 1e-20)
        y = softmax(tf.reshape(y, (-1, n_routers, n_classes)) / tau)
        return tf.reshape(y, (-1, n_routers * n_classes))

    return gumbel_softmax


def build_critic_gcn(n_nodes: int, n_features: int, dropout_rate: float) -> Model:
    """3-layer GCN backbone used inside the critic (discriminator).

    Same architecture family as the reward networks
    (`mrwgan.reward_networks.build_gcn_reward_net`) but shallower (16/32/64
    channels vs 128/64/64/32) and without the L2-regularized final layer.
    """
    from spektral.layers import GCNConv

    leaky_relu = LeakyReLU(alpha=LEAKY_RELU_ALPHA)
    l2_reg = tf.keras.regularizers.l2(2.5e-4)

    X_in = Input(shape=(n_nodes, n_features))
    A_in = Input(shape=(n_nodes, n_nodes))

    dr0 = Dropout(rate=dropout_rate)(X_in)
    x1 = GCNConv(16, activation=leaky_relu, kernel_regularizer=l2_reg)([dr0, A_in])
    dr1 = Dropout(rate=dropout_rate)(x1)
    x2 = GCNConv(32, activation=leaky_relu, kernel_regularizer=l2_reg)([dr1, A_in])
    dr2 = Dropout(rate=dropout_rate)(x2)
    x3 = GCNConv(64, activation=leaky_relu, kernel_regularizer=l2_reg)([dr2, A_in])

    return Model(inputs=[X_in, A_in], outputs=x3, name="critic_gnn")


def build_critic(n_nodes: int, n_features: int, dropout_rate: float) -> Model:
    """Full critic: GCN backbone + global sum pool + linear scalar output."""
    from spektral.layers import GlobalSumPool

    Xd_in = Input(shape=(n_nodes, n_features))
    Ad_in = Input(shape=(n_nodes, n_nodes))

    gnn = build_critic_gcn(n_nodes, n_features, dropout_rate)
    out_gnn = gnn([Xd_in, Ad_in])
    pooled = GlobalSumPool()(out_gnn)
    output = Dense(1, activation="linear")(pooled)

    return Model(inputs=[Xd_in, Ad_in], outputs=output, name="critic")


def build_generator_with_id(
    latent_dim: int, n_routers: int, n_classes: int, batch_size: int, tau: float = GUMBEL_TEMPERATURE
) -> Model:
    """Generator wrapped with Gumbel-softmax sampling + one-hot node-ID concat.

    The raw MLP output is sampled through Gumbel-softmax, reshaped to
    `(n_routers, n_classes)`, then concatenated with a one-hot router-ID
    matrix so the reward networks / critic can distinguish router positions.
    """
    size_out = n_routers * n_classes
    dim_a = (n_routers, n_routers)
    dim_x = (n_routers, n_classes)

    generator = build_generator(latent_dim, size_out)
    gumbel_softmax = make_gumbel_softmax(n_routers, n_classes, tau)

    in_noise = Input(shape=(latent_dim,))
    id_in = Input(shape=dim_a, batch_size=batch_size)

    raw = generator(in_noise)
    x_bin = Lambda(gumbel_softmax, output_shape=(n_routers * n_classes,))(raw)
    x_out = Reshape(dim_x, input_shape=(size_out,))(x_bin)
    x_out = Concatenate(axis=-1)([x_out, id_in])

    return Model([in_noise, id_in], x_out, name="generator")
