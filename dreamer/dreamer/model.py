#!/usr/bin/env python
from PIL import Image
import numpy as np
import tensorflow as tf

IMG_JITTER = 32
OCTAVE_SCALE = 1.3


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
    def __call__(
        self, img: tf.TensorSpec, steps: tf.TensorSpec, step_size: tf.TensorSpec
    ) -> tf.Tensor:
        for _ in range(steps):
            img = self._step(img, step_size)
        return img

    def _step(self, img: tf.Tensor, step_size: tf.Tensor) -> tf.Tensor:
        # Shift/offset image by random jitter
        x_shift, y_shift = np.random.randint(-IMG_JITTER, IMG_JITTER + 1, 2)
        img = tf.roll(tf.roll(img, x_shift, axis=1), y_shift, axis=0)

        loss = tf.constant(0.0)
        with tf.GradientTape() as tape:
            tape.watch(img)
            loss = activation_loss(img, self.model)

        gradients = tape.gradient(loss, img)
        # Normalize gradient steps
        gradients /= tf.math.reduce_std(gradients) + 1e-8

        img = img + gradients * step_size
        # Reverse image shift
        img = tf.roll(tf.roll(img, -x_shift, axis=1), -y_shift, axis=0)
        return tf.clip_by_value(img, -1, 1)


def preprocess_image(img: Image, max_size: int = 512) -> tf.Tensor:
    img.thumbnail((max_size, max_size))
    img = np.array(img)
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return tf.convert_to_tensor(img)


def deprocess_image(img: tf.Tensor) -> Image:
    """Reverses normalization of pixel values from image preprocessing.
    """
    img = 255 * (img + 1.0) / 2.0
    img = tf.cast(img, tf.uint8)
    img = img.numpy()
    return Image.fromarray(img)


def dream() -> None:
    model = get_dreamer_model()
    dreamer = Dreamer(model)

    img = Image.open("images/cat2.jpg")
    img = preprocess_image(img)

    img_shape = tf.shape(img)[:-1]
    img_shape_float = tf.cast(img_shape, tf.float32)
    for n in range(-2, 3):
        octave_shape = tf.cast(img_shape_float * (OCTAVE_SCALE**n), tf.int32)
        img = tf.image.resize(img, octave_shape).numpy()
        img = dreamer(img, tf.constant(25), tf.constant(0.01))

    # Resize to original preprocessed image shape
    img = tf.image.resize(img, img_shape)
    img = deprocess_image(img)
    img.show()


if __name__ == "__main__":
    dream()
