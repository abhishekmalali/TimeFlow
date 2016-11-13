import tensorflow as tf


def RMSE(predicted_tensor, response_tensor):

    """Initialize InputLSTMLayer class

    Parameters
    ----------
    predicted_tensor : tf.Tensor
        Predicted tensor
    response_tensor: tf.Tensor
        Response tensor

    Response
    ----------
    tf.Tensor
        RMSE operation tensor
    """
    return tf.reduce_mean(tf.pow(tf.sub(predicted_tensor, response_tensor), 2))


def R2(predicted_tensor, response_tensor):

    """Initialize InputLSTMLayer class

    Parameters
    ----------
    predicted_tensor : tf.Tensor
        Predicted tensor
    response_tensor: tf.Tensor
        Response tensor

    Response
    ----------
    tf.Tensor
        R2 operation tensor
    """
    pred_mean = tf.expand_dims(tf.reduce_mean(predicted_tensor,
                               reduction_indices=[0]), 1)
    rse = tf.reduce_mean(tf.pow(tf.sub(predicted_tensor, pred_mean), 2))
    mse = tf.reduce_mean(tf.pow(tf.sub(predicted_tensor, response_tensor), 2))
    R2 = 1 - tf.div(mse, rse)
    return R2
