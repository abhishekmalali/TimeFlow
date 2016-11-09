# Input and output placeholders for prediction placeholders
import tensorflow as tf


def input_placeholder(input_dim, name='inputs'):
    """Initialize input placeholder for prediction networks

    Parameters
    ----------
    input_dim : integer
        Input dimensions
    name : string
        Placeholder name

    Returns
    ----------
    tf.placeholder
        Input placeholder

    """
    input = tf.placeholder(tf.float32, shape=[None, input_dim],
                           name=name)
    return input


def output_placeholder(output_dim, name='outputs'):
    """Initialize output placeholder for prediction networks

    Parameters
    ----------
    output_dim : integer
        Input dimensions
    name : string
        Placeholder name

    Returns
    ----------
    tf.placeholder
        Input placeholder

    """
    output = tf.placeholder(tf.float32, shape=[None, output_dim],
                            name=name)
    return output
