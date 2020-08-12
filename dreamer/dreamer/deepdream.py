#!/usr/bin/env python
import argparse
import math
from typing import List
from typing import Tuple

from PIL import Image
import numpy as np
import tensorflow as tf

IMG_JITTER = 128
VALID_OUTPUT_LAYERS = frozenset(f"mixed{i}" for i in range(11))


def get_deep_dream_model(output_layers: List[str]) -> tf.keras.Model:
    if not all(l in VALID_OUTPUT_LAYERS for l in output_layers):
        raise ValueError(
            f"Valid output layers for InceptionV3 are {VALID_OUTPUT_LAYERS}"
        )

    base_model = tf.keras.applications.InceptionV3(
        include_top=False, weights="imagenet"
    )
    layers = [base_model.get_layer(layer_name).output for layer_name in output_layers]
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
            tf.TensorSpec(shape=[], dtype=tf.int32),
        )
    )
    def __call__(self, img, steps, step_size, tile_size) -> tf.Tensor:
        img_shape = tf.shape(img)
        width = img_shape[0]
        height = img_shape[1]
        tile_xs = tf.range(0, width, tile_size)
        tile_ys = tf.range(0, height, tile_size)

        for _ in range(steps):
            gradients = tf.zeros_like(img)

            # Shift/offset image by random jitter
            x_shift, y_shift = np.random.randint(-IMG_JITTER, IMG_JITTER + 1, 2)
            img = tf.roll(tf.roll(img, x_shift, axis=1), y_shift, axis=0)

            # Iterate over all image tiles for each step
            for x in tile_xs:
                for y in tile_ys:
                    # Skip any tiles that may be too small
                    if x + tile_size > width or y + tile_size > height:
                        continue
                    with tf.GradientTape() as tape:
                        tape.watch(img)
                        img_tile = img[
                            x : x + tile_size,
                            y : y + tile_size,
                        ]
                        loss = activation_loss(img_tile, self.model)
                    gradients += tape.gradient(loss, img)

            # Normalize gradient steps
            gradients /= tf.math.reduce_std(gradients) + 1e-8
            img = img + gradients * step_size
            img = tf.clip_by_value(img, -1, 1)

            # Reverse image shift
            img = tf.roll(tf.roll(img, -x_shift, axis=1), -y_shift, axis=0)

        return img


def get_random_image(shape: Tuple[int, int]) -> Image:
    """Generates an image of the given shape from random noise.
    """
    width, height = shape
    img_array = np.random.randint(0, 256, (width, height, 3), dtype="uint8")
    return Image.fromarray(img_array)


def get_image(image_filepath: str) -> Image:
    if image_filepath == "random":
        return get_random_image((512, 512))
    return Image.open(image_filepath)


def preprocess_image(input_img: Image, max_size: int = 1024) -> tf.Tensor:
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


def dream(
    input_img: Image,
    output_layers: List[str],
    octaves: int,
    octave_scale: float,
    steps: int,
    step_size: float,
    tile_size: int,
) -> Image:
    model = get_deep_dream_model(output_layers)
    dreamer = DeepDream(model)

    img = preprocess_image(input_img)
    img_shape = tf.shape(img)[:-1]
    img_shape_float = tf.cast(img_shape, tf.float32)
    octave_range = range(-math.floor(octaves / 2), math.ceil(octaves / 2))
    for i in octave_range:
        octave_shape = tf.cast(img_shape_float * (octave_scale ** i), tf.int32)
        img = tf.image.resize(img, octave_shape).numpy()
        img = dreamer(
            img, tf.constant(steps), tf.constant(step_size), tf.constant(tile_size)
        )

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
        "-i",
        "--image-filepath",
        type=str,
        default="random",
        help="Filepath for an input image. Uses random noise by default.",
    )
    parser.add_argument(
        "-o",
        "--output-filepath",
        type=str,
        help="Filepath to save the output image to.",
        default="images/dream.png",
    )
    parser.add_argument(
        "--octaves",
        type=int,
        default=5,
        help="Number of 'octaves' or image scales to use in generating output image.",
    )
    parser.add_argument(
        "--octave-scale",
        type=float,
        default=1.3,
        help="Value to scale each 'octave' image by.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=25,
        help="Number of steps or iterations to use for each image 'octave'.",
    )
    parser.add_argument(
        "--step-size",
        type=float,
        default=0.01,
        help="Value to scale each step change by during image generation.",
    )
    parser.add_argument(
        "--output-layers",
        type=str,
        nargs="+",
        default=["mixed3", "mixed5"],
        help="The names of layers in the InceptionV3 model to use as output layers.",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=256,
        help="Size of image tiles, in pixels. Each image will be broken into tiles and each tile passed to the model separately.",
    )
    args = parser.parse_args()
    input_img = get_image(args.image_filepath)
    output_img = dream(
        input_img,
        args.output_layers,
        args.octaves,
        args.octave_scale,
        args.steps,
        args.step_size,
        args.tile_size,
    )
    output_img.save(args.output_filepath)
