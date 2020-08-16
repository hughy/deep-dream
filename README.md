# DeepDream
DeepDream was an experiment that Google engineer Alexander Mordvintsev implemented in 2015 to visualize the inner representations of deep neural networks. Mordvintsev's [blog post](https://ai.googleblog.com/2015/06/inceptionism-going-deeper-into-neural.html) describing this experiment received a lot of media attention at the time. This project uses TensorFlow to imeplement a DeepDream image generator.
## Examples

<div style="display: inline">
    <img src="images/cat.jpg" width="256" />
    <img src="images/dream.png" width="256" />
</div>
<div style="display: inline">
    <img src="images/half_dome.jpg" width="256" />
    <img src="images/half_dome_dream.png" width="256" />
</div>

## Implementation

I followed the [TensorFlow DeepDream tutorial](https://www.tensorflow.org/tutorials/generative/deepdream) in implementing DeepDream and added features from the notebook in the [deepdream repo](https://github.com/google/deepdream/blob/master/dream.ipynb).

### Command-line options

    poetry run python -m dreamer.deepdream --help                                
    usage: deepdream.py [-h] [-i IMAGE_FILEPATH] [-o OUTPUT_FILEPATH] [--octaves OCTAVES] [--octave-scale OCTAVE_SCALE]
                        [--steps STEPS] [--step-size STEP_SIZE] [--output-layers OUTPUT_LAYERS [OUTPUT_LAYERS ...]]
    
    Generates 'dream-like' variations of an input image using a minimal DeepDream implementation
    (https://ai.googleblog.com/2015/06/inceptionism-going-deeper-into-neural.html)
    
    optional arguments:
      -h, --help            show this help message and exit
      -i IMAGE_FILEPATH, --image-filepath IMAGE_FILEPATH
                            Filepath for an input image. Uses random noise by default.
      -o OUTPUT_FILEPATH, --output-filepath OUTPUT_FILEPATH
                            Filepath to save the output image to.
      --octaves OCTAVES     Number of 'octaves' or image scales to use in generating output image.
      --octave-scale OCTAVE_SCALE
                            Value to scale each 'octave' image by.
      --steps STEPS         Number of steps or iterations to use for each image 'octave'.
      --step-size STEP_SIZE
                            Value to scale each step change by during image generation.
      --output-layers OUTPUT_LAYERS [OUTPUT_LAYERS ...]
                            The names of layers in the InceptionV3 model to use as output layers.
      --tile-size TILE_SIZE
                            Size of image tiles, in pixels. Each image will be broken into tiles and each tile passed to the model
                            separately.
    
### Octaves
Applying gradient ascent to the input image at different scales or 'octaves' allows DeepDream to generate images that feature patterns at different levels of granularity. The TensorFlow tutorial explains octaves with the following:

> ...applying gradient ascent at different scales. This will allow patterns generated at smaller scales to be incorporated into patterns at higher scales and filled in with additional detail.

### Jitter
Each step in the image generation process applies random jitter to shift the image before feeding it to the network. This use of jitter follows the implementation in the Deep Dreams notebook.

## Future work
I'd like to extend this project to support additional pretrained networks
and/or different pretrained weights for InceptionV3. The original DeepDream blog post features some examples that used a network trained on the MIT places dataset. MIT's CSAIL provides a repository of Caffe models pretrained on the places dataset (https://github.com/CSAILVision/places365). Converting the wieghts from one of these networks to a TensorFlow-compatible format seems like the simplest path to extending this project to use the places dataset.

In the future I would also like to add an implementation of DeepDream using
Tensorflow.js. Running DeepDream in the browser may facilitate more interesting
visualization and rendering opportunities.
