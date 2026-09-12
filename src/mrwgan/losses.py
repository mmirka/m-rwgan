"""WGAN-GP loss machinery and the adversarial/reward loss-weight annealing schedule.

Contains the gradient-penalty layer, the Wasserstein loss, and the
per-reward MSE losses gated by the shared ``LAMBDA`` annealing variable.

The ``GradientPenalty`` layer below computes its term with ``tf.gradients``,
which only connects a gradient in graph mode — see ``mrwgan.training`` for
why this stack runs graph-mode.
"""
from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import backend as K


class RandomWeightedAverage(tf.keras.layers.Layer):
    """Interpolates real and fake samples for the WGAN-GP gradient penalty."""

    def __init__(self, batch_size: int, **kwargs):
        super().__init__(**kwargs)
        self.batch_size = batch_size

    def call(self, inputs, **kwargs):
        alpha = tf.random.uniform((self.batch_size, 1, 1))
        return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])

    def compute_output_shape(self, input_shape):
        return input_shape[0]


class GradientPenalty(tf.keras.layers.Layer):
    """Computes ``||grad(critic(interpolated))|| - 1`` for the WGAN-GP penalty term."""

    def call(self, inputs):
        target, wrt = inputs
        grad = tf.gradients(target, wrt)[0]
        return K.sqrt(K.sum(K.batch_flatten(K.square(grad)), axis=1, keepdims=True)) - 1

    def compute_output_shape(self, input_shapes):
        return (input_shapes[1][0], 1)


def wasserstein_loss(y_true, y_pred):
    return tf.reduce_mean(y_true * y_pred)


class LambdaSchedule:
    """Anneals the adversarial/reward loss balance over training.

    `LAMBDA` starts at 1 (all-adversarial) and decays by `delta` every
    `every_n_epochs` epochs once `start_epoch` is reached, floored at
    `floor`. Reward losses are scaled by `(1 - LAMBDA)`, the adversarial
    loss by `LAMBDA` — so training shifts weight from "look real" to
    "match the target rewards" as it progresses. Defaults: `delta = -0.05`,
    starting at epoch 50, every 10 epochs, floor 0.2.
    """

    def __init__(
        self,
        initial: float = 1.0,
        delta: float = -0.05,
        start_epoch: int = 50,
        every_n_epochs: int = 10,
        floor: float = 0.2,
    ):
        self.value = tf.Variable(initial, dtype="float32", name="LAMBDA")
        if not tf.executing_eagerly():
            # Graph mode (TF>=2.4 needs it for the WGAN-GP gradient penalty,
            # see mrwgan.training): this bare tf.Variable is outside Keras'
            # tracking, so nothing else will initialise it in the backend
            # session before train_on_batch reads it.
            tf.compat.v1.keras.backend.get_session().run(self.value.initializer)
        self.delta = delta
        self.start_epoch = start_epoch
        self.every_n_epochs = every_n_epochs
        self.floor = floor
        self._current = initial

    def step(self, epoch: int) -> None:
        if epoch >= self.start_epoch and epoch % self.every_n_epochs == 0 and self._current > self.floor:
            self._current += self.delta
            K.set_value(self.value, self._current)


def make_reward_loss(lambda_schedule: LambdaSchedule):
    """MSE reward loss scaled by `(1 - LAMBDA)`. One instance per reward head."""

    def reward_loss(y_true, y_pred):
        mse = K.mean(K.sum(K.square(y_true - y_pred)))
        return (1 - lambda_schedule.value) * tf.cast(mse, tf.float32)

    return reward_loss


def make_generator_wasserstein_loss(lambda_schedule: LambdaSchedule):
    """Adversarial generator loss scaled by `LAMBDA`."""

    def g_wasserstein_loss(y_true, y_pred):
        return lambda_schedule.value * K.mean(y_true * y_pred)

    return g_wasserstein_loss


def crop_channels(start: int, end: int):
    """`Lambda`-compatible layer factory: crop the last axis to `[start:end]`.

    Used to strip the one-hot router-ID channels before feeding the
    generator's output to the CNN (area) reward network, which expects only
    the `n_classes` router-class channels.
    """
    return tf.keras.layers.Lambda(lambda x: x[:, :, :, start:end])
