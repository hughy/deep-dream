#!/usr/bin/env python
import argparse

from PIL import Image
import numpy as np
import tensorflow as tf

IMG_JITTER = 64
OCTAVE_SCALE = 1.3


def get_deep_dream_model() -> tf.keras.Model:
    base_model = tf.keras.applications.InceptionV3(
        include_top=False, weights="imagenet"
    )
    layers = [
        base_model.get_layer(layer_name).output for layer_name in ("mixed3", "mixed5")
    ]
    return tf.keras.Model(inputs=base_model.input, outputs=layers)


def activation_loss(img: tf.Tensor, model: tf.keras.Model) -> tf.Tensor:
    """Calculates loss as the sum of the activations of the output layers of the model.
    """
    img_batch = tf.expand_dims(img, axis=0)
    layer_activations = model(img_batch)
    if len(layer_activations) == 1:
        layer_activations = [layer_activations]

    return tf.reduce_sum(
        [tf.math.reduce_mean(activations) for activations in layer_activations]
    )


class DeepDream(tf.Module):
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
    def __call__(self, img, steps, step_size) -> tf.Tensor:
        for _ in range(steps):
            img = self._step(img, step_size)
        return img

    @tf.function(
        input_signature=(
            tf.TensorSpec(shape=[None, None, 3], dtype=tf.float32),
            tf.TensorSpec(shape=[], dtype=tf.float32),
        )
    )
    def _step(self, img, step_size) -> tf.Tensor:
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


def preprocess_image(input_img: Image, max_size: int = 512) -> tf.Tensor:
    input_img.thumbnail((max_size, max_size))
    img_array = np.array(input_img)
    img_array = tf.keras.applications.inception_v3.preprocess_input(img_array)
    return tf.convert_to_tensor(img_array)


def deprocess_image(img: tf.Tensor) -> Image:
    """Reverses normalization of pixel values from image preprocessing.
    """
    img = 255 * (img + 1.0) / 2.0
    img = tf.cast(img, tf.uint8)
    return Image.fromarray(img.numpy())


def dream(image_filepath: str) -> Image:
    model = get_deep_dream_model()
    dreamer = DeepDream(model)

    input_img = Image.open(image_filepath)
    img = preprocess_image(input_img)

    img_shape = tf.shape(img)[:-1]
    img_shape_float = tf.cast(img_shape, tf.float32)
    # Iterate over five 'octaves'
    for i in range(-2, 3):
        octave_shape = tf.cast(img_shape_float * (OCTAVE_SCALE ** i), tf.int32)
        img = tf.image.resize(img, octave_shape).numpy()
        img = dreamer(img, tf.constant(25), tf.constant(0.01))

    # Resize to original preprocessed image shape
    img = tf.image.resize(img, img_shape)
    return deprocess_image(img)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""
        Generates 'dream-like' variations of an input image using a minimal DeepDream
        implementation (https://ai.googleblog.com/2015/06/inceptionism-going-deeper-into-neural.html)
    """.strip()
    )
    parser.add_argument(
        "--image-filepath",
        type=str,
        help="Filepath for an input image.",
        default="images/cat.jpg",
    )
    args = parser.parse_args()
    output_img = dream(args.image_filepath)
    output_img.save("images/dream.png")
