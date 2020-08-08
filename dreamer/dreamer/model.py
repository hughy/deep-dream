#!/usr/bin/env python
from PIL import Image
import numpy as np
import tensorflow as tf


def get_dreamer_model() -> tf.keras.Model:
    base_model = tf.keras.applications.InceptionV3(
        include_top=False, weights="imagenet"
    )
    layers = [
        base_model.get_layer(layer_name).output for layer_name in ("mixed3", "mixed5")
    ]
    return tf.keras.Model(inputs=base_model.input, outputs=layers)


def activation_loss(img: tf.Tensor, model: tf.keras.Model):
    """Calculates loss as the sum of the activations of the output layers of the model.
    """
    img_batch = tf.expand_dims(img, axis=0)
    layer_activations = model(img_batch)
    if len(layer_activations) == 1:
        layer_activations = [layer_activations]

    return tf.reduce_sum(
        [tf.math.reduce_mean(activations) for activations in layer_activations]
    )


class Dreamer(tf.Module):
    def __init__(self, model: tf.keras.Model) -> None:
        super().__init__()
        self.model = model

    @tf.function(
        input_signature=(
            tf.TensorSpec(shape=[None, None, 3], dtype=tf.float32),
            tf.TensorSpec(shape=[], dtype=tf.int32),
            tf.TensorSpec(shape=[], dtype=tf.float32),
        )
    )
    def __call__(self, img: tf.Tensor, steps: tf.Tensor, step_size: tf.Tensor) -> tf.Tensor:
        for _ in range(steps):
            img = self._step(img, step_size)
        return img

    def _step(self, img: tf.Tensor, step_size: tf.Tensor) -> tf.Tensor:
        loss = tf.constant(0.0)
        with tf.GradientTape() as tape:
            tape.watch(img)
            loss = activation_loss(img, self.model)

        gradients = tape.gradient(loss, img)
        gradients /= tf.math.reduce_std(gradients) + 1e-8

        img = img + gradients * step_size
        return tf.clip_by_value(img, -1, 1)


def preprocess_image(img: Image) -> tf.Tensor:
    img = np.array(img)
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return tf.convert_to_tensor(img)


def deprocess_image(img: tf.Tensor) -> Image:
    img = 255 * (img + 1.0) / 2.0
    return tf.cast(img, tf.uint8)


if __name__ == "__main__":
    model = get_dreamer_model()
    dreamer = Dreamer(model)

    img = Image.open("images/cat0.jpg")
    img = preprocess_image(img)

    output_img = dreamer(img, tf.constant(100), tf.constant(0.01))
    output_img = deprocess_image(output_img)
    output_img = Image.fromarray(np.array(output_img))
    output_img.show()
