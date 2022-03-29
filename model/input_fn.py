"""Create the input data pipeline using tf.data"""

import tensorflow as tf


def _parse_function(filename, label, size):
    """Obtain the image from the filename
    The following operations are applied:
        - Decode the image from jpeg
        - Conver to float and to range [0, 1]
    """
    
    # Read file and decode jpeg image
    image_string = tf.io.read_file(filename)
    image_decode = tf.image.decode_jpeg(image_string)

    # Convert tensor to float
    image = tf.image.convert_image_dtype(image_decode, dtype=tf.float32)

    # resize image
    resized_image = tf.image.resize(image, [size, size])

    # convert label datatype to int
    label = tf.strings.to_number(label, tf.int64)
    return resized_image, label


def train_preprocesses(image, label, use_random_flip):
    """Image preprocessing for training
    Apply following operations:
        - Horizontally flip the image with probability 1/2
        - Apply random brightness and saturation  
    """

    if use_random_flip:
        image = tf.image.random_flip_left_right(image)
    
    image = tf.image.random_brightness(image, max_delta = 32.0 / 255.0)
    image = tf.image.random_saturation(image, lower=0.5, upper=1.5)

    # Make sure the image is still in [0, 1]
    image = tf.clip_by_value(image, 0.0, 1.0)

    return image, label



def input_fn(is_training, filenames, labels, params):
    """Input function for SIGNS dataset.
    
    Args:
        is_training: (bool) whether to use train or test pipelines
        filenames: (list) filenames of image as ["data_dir/{label}_IMG_{id}.jpg",...]
        labels: (list) corresponding list of labels
        params: (Params) contains hyperparameters of the model
    """
    num_samples = len(filenames)
    assert len(filenames) == len(labels), "Filenames and labesl should have same length"

    # Create a dataset serving batches of images and labels
    parse_fn = lambda f, l: _parse_function(f, l, params.image_size)
    train_fn = lambda f, l: train_preprocesses(f, l, params.use_random_flip)

    if is_training:
        dataset = (tf.data.Dataset.from_tensor_slices((tf.constant(filenames), tf.constant(labels)))
            .shuffle(num_samples)
            .map(parse_fn, num_parallel_calls=params.num_parallel_calls)
            .map(train_fn, num_parallel_calls=params.num_parallel_calls)
            .batch(params.batch_size)
            .prefetch(1))
    else:
        dataset = (tf.data.Dataset.from_tensor_slices((tf.constant(filenames), tf.constant(labels)))
            .map(parse_fn, num_parallel_calls=params.num_parallel_calls)
            .batch(params.batch_size)
            .prefetch(1))
    
    # iterator = iter(dataset)
    # images, labels = next(iterator)
    # inputs = {'images': images, 'labels': labels, 'iterator': iterator}
    return dataset
        
