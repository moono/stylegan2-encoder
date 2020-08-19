# import numpy as np
import tensorflow as tf

# from PIL import Image


# tensorflow related
def allow_memory_growth():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
    return


def adjust_dynamic_range(images, range_in, range_out, out_dtype):
    scale = (range_out[1] - range_out[0]) / (range_in[1] - range_in[0])
    bias = range_out[0] - range_in[0] * scale
    images = images * scale + bias
    images = tf.clip_by_value(images, range_out[0], range_out[1])
    images = tf.cast(images, dtype=out_dtype)
    return images


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')
