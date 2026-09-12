"""M-RWGAN training loop.

Trains the reward-weighted WGAN-GP generator and saves checkpoints/history.

Training data requirement
--------------------------
Provide a pickled list of :class:`mrwgan.data_types.Data_AX` objects for
your 8x8 mesh (see ``data/raw/README.md`` for the datasets used in the
paper). The critic only ever sees router-*class* assignments (`X`),
GCN-normalized and concatenated with a one-hot node-ID (via
`mrwgan.data_types.build_gcn_dataset`, the `GCNFilter` + `nodeID`
pipeline); the mesh adjacency matrix is fixed and identical for every
sample, taken from the first entry of the dataset.
"""
from __future__ import annotations

import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import scipy.sparse
import tensorflow as tf
from sklearn.utils import shuffle
from tensorflow.keras.layers import Input, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop

# The WGAN-GP gradient-penalty layer (mrwgan.losses.GradientPenalty) computes its
# term with tf.gradients(), which only connects a gradient in graph mode. TF >= 2.4
# builds Keras functional models eagerly, where tf.gradients() returns None
# ("None values not supported") at model-construction time. Disable eager
# execution for this process so the functional model builds unchanged. Training
# here never needs eager execution (no custom train loop, just compiled
# model.train_on_batch).
if tf.executing_eagerly():
    tf.compat.v1.disable_eager_execution()

from .data_types import build_gcn_dataset, load_data_ax_dataset  # noqa: E402
from .gpu import configure_gpu  # noqa: E402
from .losses import (  # noqa: E402
    GradientPenalty,
    LambdaSchedule,
    RandomWeightedAverage,
    crop_channels,
    make_generator_wasserstein_loss,
    make_reward_loss,
    wasserstein_loss,
)
from .model import build_critic, build_generator_with_id  # noqa: E402
from .reward_networks import load_reward_network  # noqa: E402


@dataclass
class TrainConfig:
    dataset_path: str
    reward_sat_checkpoint: str
    reward_pow_checkpoint: str
    reward_area_checkpoint: str
    output_dir: str
    reward_weights: tuple[float, float, float] = (0.5, 0.1, 0.4)
    epochs: int = 300
    batch_size: int = 16
    latent_dim: int = 100
    mesh_rows: int = 8
    mesh_cols: int = 8
    nb_classes: int = 3
    dropout_rate: float = 0.1
    critic_lr: float = 5e-5
    beta: float = 10.0
    gradient_penalty_weight: float = 10.0
    seed: int | None = None
    fixed_noise_samples: int = 10
    eval_samples: int = 100
    run_name: str = field(default="")


def _load_real_dataset(config: TrainConfig):
    """Load the real router-class dataset through the same GCN-normalization +
    node-ID pipeline used everywhere else (critic, reward nets): see
    `mrwgan.data_types.build_gcn_dataset`. Returns `(real_x, adjacency)` where
    `real_x` has shape `(n_samples, n_routers, nb_classes + n_routers)` and
    `adjacency` is the GCN-normalized `(n_routers, n_routers)` matrix shared
    by every sample (all samples use the same fixed 8x8-mesh topology).
    """
    raw = load_data_ax_dataset(config.dataset_path)

    n_routers = config.mesh_rows * config.mesh_cols
    dataset = build_gcn_dataset(raw, config.nb_classes, max_id=n_routers - 1)
    real_x = np.asarray([g.x for g in dataset], dtype=np.float32)

    adjacency = dataset[0].a
    if scipy.sparse.issparse(adjacency):
        adjacency = adjacency.toarray()
    adjacency = np.asarray(adjacency, dtype=np.float32)
    return real_x, adjacency


def build_models(config: TrainConfig, n_features: int):
    """Construct critic, generator, and the two compiled training models
    (discriminator_model / generator_model)."""
    n_routers = config.mesh_rows * config.mesh_cols
    dim_a = (n_routers, n_routers)
    dim_real_x = (n_routers, n_features)

    critic = build_critic(n_routers, n_features, config.dropout_rate)
    generator = build_generator_with_id(
        config.latent_dim, n_routers, config.nb_classes, config.batch_size
    )

    reward_sat = load_reward_network(config.reward_sat_checkpoint, name="R_sat")
    reward_pow = load_reward_network(config.reward_pow_checkpoint, name="R_pow")
    reward_area = load_reward_network(config.reward_area_checkpoint, name="R_area")

    lambda_schedule = LambdaSchedule()

    # ---- discriminator_model: critic trained via WGAN-GP ----
    critic.trainable = True
    generator.trainable = False

    z_disc = Input(shape=(config.latent_dim,))
    id_in_d = Input(shape=dim_a, batch_size=config.batch_size)
    a_cst_d = Input(shape=dim_a, batch_size=config.batch_size)
    real_x = Input(shape=dim_real_x, batch_size=config.batch_size)

    fake_x = generator([z_disc, id_in_d])
    fake_out = critic([fake_x, a_cst_d])
    real_out = critic([real_x, a_cst_d])

    averaged = RandomWeightedAverage(config.batch_size)([real_x, fake_x])
    # training=False so the critic's Dropout layers are identities on the
    # gradient-penalty path. Otherwise Dropout lowers to a tf.cond and the
    # penalty's second-order gradient (RMSprop differentiating GradientPenalty,
    # which itself differentiates the critic) fails to connect through it under
    # TF >= 2.4 graph mode. Measuring the Lipschitz penalty without dropout
    # noise is also the standard WGAN-GP formulation.
    validity_averaged = critic([averaged, a_cst_d], training=False)
    gp = GradientPenalty()([validity_averaged, averaged])

    discriminator_model = Model(
        inputs=[real_x, z_disc, a_cst_d, id_in_d],
        outputs=[real_out, fake_out, gp],
        name="discriminator_model",
    )
    discriminator_model.compile(
        loss=[wasserstein_loss, wasserstein_loss, "mse"],
        optimizer=RMSprop(learning_rate=config.critic_lr),
    )
    discriminator_model.summary()

    # ---- generator_model: generator trained against critic + 3 rewards ----
    generator.trainable = True
    critic.trainable = False

    z_gen = Input(shape=(config.latent_dim,))
    id_in_g = Input(shape=dim_a, batch_size=config.batch_size)
    a_cst_g = Input(shape=dim_a, batch_size=config.batch_size)

    x_gen = generator([z_gen, id_in_g])
    valid = critic([x_gen, a_cst_g])
    rwd_sat = reward_sat([x_gen, a_cst_g])
    rwd_pow = reward_pow([x_gen, a_cst_g])

    x_gen_grid = Reshape((config.mesh_rows, config.mesh_cols, n_features))(x_gen)
    x_gen_grid_no_id = crop_channels(0, config.nb_classes)(x_gen_grid)
    rwd_area = reward_area([x_gen_grid_no_id])

    generator_model = Model(
        inputs=[z_gen, a_cst_g, id_in_g],
        outputs=[valid, rwd_sat, rwd_pow, rwd_area],
        name="generator_model",
    )

    t_sat, t_pow, t_area = config.reward_weights
    loss_weights = [1, config.beta * t_sat, config.beta * t_pow, config.beta * t_area]
    generator_model.compile(
        loss=[
            make_generator_wasserstein_loss(lambda_schedule),
            make_reward_loss(lambda_schedule),
            make_reward_loss(lambda_schedule),
            make_reward_loss(lambda_schedule),
        ],
        optimizer=RMSprop(learning_rate=config.critic_lr),
        loss_weights=loss_weights,
    )
    generator_model.summary()

    return {
        "critic": critic,
        "generator": generator,
        "discriminator_model": discriminator_model,
        "generator_model": generator_model,
        "reward_sat": reward_sat,
        "reward_pow": reward_pow,
        "reward_area": reward_area,
        "lambda_schedule": lambda_schedule,
    }


def train(config: TrainConfig) -> dict:
    """Run the full M-RWGAN training loop and save checkpoints + history.

    Returns a dict of the artifact paths written under `config.output_dir`.
    """
    # Enable GPU memory growth / report the device for callers that invoke
    # train() directly. The CLI (scripts/train_mrwgan.py) already prints its own
    # configure_gpu() summary; a second call here is harmless.
    print(configure_gpu())

    if config.seed is not None:
        np.random.seed(config.seed)
        tf.random.set_seed(config.seed)

    dataset_path = Path(config.dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Training dataset not found: {dataset_path}\n"
            "Provide a pickled list of mrwgan.data_types.Data_AX objects for "
            "your 8x8 mesh (see data/raw/README.md for the datasets used in "
            "the paper)."
        )

    real_x, adjacency = _load_real_dataset(config)
    n_routers = config.mesh_rows * config.mesh_cols
    n_features = real_x.shape[-1]  # nb_classes + n_routers (one-hot node ID, see build_gcn_dataset)

    models = build_models(config, n_features)
    critic = models["critic"]
    generator = models["generator"]
    discriminator_model = models["discriminator_model"]
    generator_model = models["generator_model"]
    reward_sat, reward_pow, reward_area = (
        models["reward_sat"],
        models["reward_pow"],
        models["reward_area"],
    )
    lambda_schedule = models["lambda_schedule"]

    batch_size = config.batch_size
    real_goal = np.ones(batch_size)
    fake_goal = -np.ones(batch_size)
    dummy = np.zeros(batch_size)
    goal_reward = np.ones(batch_size, dtype="float32")

    identity_batch = np.zeros((batch_size, n_routers, n_routers), dtype=np.float32)
    for i in range(batch_size):
        identity_batch[i] = np.eye(n_routers)
    adjacency_batch = np.broadcast_to(adjacency, (batch_size, n_routers, n_routers)).astype(np.float32)

    identity_eval = np.zeros((config.eval_samples, n_routers, n_routers), dtype=np.float32)
    for i in range(config.eval_samples):
        identity_eval[i] = np.eye(n_routers)
    adjacency_eval = np.broadcast_to(
        adjacency, (config.eval_samples, n_routers, n_routers)
    ).astype(np.float32)

    identity_fixed = np.zeros(
        (config.fixed_noise_samples, n_routers, n_routers), dtype=np.float32
    )
    for i in range(config.fixed_noise_samples):
        identity_fixed[i] = np.eye(n_routers)
    noise_fixed = np.random.rand(config.fixed_noise_samples, config.latent_dim).astype(np.float32)

    generator_loss_hist, discriminator_loss_hist = [], []
    reward_sat_hist, reward_pow_hist, reward_area_hist = [], [], []
    history_fake_fix = []

    start_train = time.time()
    for epoch in range(config.epochs):
        lambda_schedule.step(epoch)

        real_x = shuffle(real_x, random_state=config.seed)
        n_batches = len(real_x) // batch_size

        noise_eval = np.random.rand(config.eval_samples, config.latent_dim).astype(np.float32)
        gene_eval_raw = generator.predict([noise_eval, identity_eval])
        gene_eval_grid = gene_eval_raw.reshape(
            (config.eval_samples, config.mesh_rows, config.mesh_cols, n_features)
        )
        gene_eval_no_id = gene_eval_grid[:, :, :, : config.nb_classes]

        gene_fix_raw = generator.predict([noise_fixed, identity_fixed])
        gene_fix_grid = gene_fix_raw.reshape(
            (config.fixed_noise_samples, config.mesh_rows, config.mesh_cols, n_features)
        )
        history_fake_fix.append(gene_fix_grid[:, :, :, : config.nb_classes])

        r_sat = float(np.mean(reward_sat.predict([gene_eval_raw, adjacency_eval])))
        r_pow = float(np.mean(reward_pow.predict([gene_eval_raw, adjacency_eval])))
        r_area = float(np.mean(reward_area.predict([gene_eval_no_id])))
        reward_sat_hist.append(r_sat)
        reward_pow_hist.append(r_pow)
        reward_area_hist.append(r_area)

        for i in range(n_batches):
            noise = np.random.rand(batch_size, config.latent_dim).astype(np.float32)
            real_batch = real_x[i * batch_size : (i + 1) * batch_size]
            d_loss = discriminator_model.train_on_batch(
                [real_batch, noise, adjacency_batch, identity_batch],
                [real_goal, fake_goal, dummy],
            )
            discriminator_loss_hist.append(d_loss)

            noise = np.random.rand(batch_size, config.latent_dim).astype(np.float32)
            g_loss = generator_model.train_on_batch(
                [noise, adjacency_batch, identity_batch],
                [real_goal, goal_reward, goal_reward, goal_reward],
            )
            generator_loss_hist.append(g_loss)

        print(
            f"epoch {epoch:4d}/{config.epochs}  "
            f"D={d_loss[0]:.4f}  G={g_loss[0]:.4f}  "
            f"R_sat={r_sat:.4f}  R_pow={r_pow:.4f}  R_area={r_area:.4f}"
        )

    training_time = time.time() - start_train
    print(f"Training complete in {training_time:.1f}s")

    return _save_artifacts(
        config,
        generator_model=generator_model,
        generator=generator,
        critic=critic,
        history_fake_fix=history_fake_fix,
        generator_loss_hist=generator_loss_hist,
        discriminator_loss_hist=discriminator_loss_hist,
        reward_hists=(reward_sat_hist, reward_pow_hist, reward_area_hist),
        training_time=training_time,
    )


def _save_artifacts(
    config: TrainConfig,
    *,
    generator_model,
    generator,
    critic,
    history_fake_fix,
    generator_loss_hist,
    discriminator_loss_hist,
    reward_hists,
    training_time,
) -> dict:
    t_sat, t_pow, t_area = config.reward_weights
    run_name = config.run_name or (
        f"mrwgan_sat{int(t_sat * 100)}_pow{int(t_pow * 100)}_area{int(t_area * 100)}"
    )

    checkpoint_dir = Path(config.output_dir) / "checkpoints" / run_name
    log_dir = Path(config.output_dir) / "logs" / run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    generator_path = checkpoint_dir / "generator.h5"
    critic_path = checkpoint_dir / "critic.h5"
    generator.save(generator_path)
    critic.save(critic_path)

    reward_sat_hist, reward_pow_hist, reward_area_hist = reward_hists
    artifacts = {
        "generator_checkpoint": str(generator_path),
        "critic_checkpoint": str(critic_path),
        "history_fake_fix": str(log_dir / "history_fake_fix.pkl"),
        "generator_loss": str(log_dir / "generator_loss.pkl"),
        "discriminator_loss": str(log_dir / "discriminator_loss.pkl"),
        "reward_sat_loss": str(log_dir / "reward_sat_loss.pkl"),
        "reward_pow_loss": str(log_dir / "reward_pow_loss.pkl"),
        "reward_area_loss": str(log_dir / "reward_area_loss.pkl"),
        "training_time_seconds": training_time,
    }

    for key, values in [
        ("history_fake_fix", history_fake_fix),
        ("generator_loss", np.asarray(generator_loss_hist, dtype=object)),
        ("discriminator_loss", np.asarray(discriminator_loss_hist, dtype=object)),
        ("reward_sat_loss", np.asarray(reward_sat_hist)),
        ("reward_pow_loss", np.asarray(reward_pow_hist)),
        ("reward_area_loss", np.asarray(reward_area_hist)),
    ]:
        with open(artifacts[key], "wb") as f:
            pickle.dump(values, f)

    return artifacts
