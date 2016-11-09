from .nn_layer import NNLayer
import tensorflow as tf

__all__ = ['InputLSTMLayer']


class InputLSTMLayer(NNLayer):
    """
    This layer expands dimensions inorder for the LSTM to function.
    The input data has a shape of (?,input_dimensions) and the output
    for this layer is a tensor of the shape (1, ?, input_dimensions)
    """
    def __init__(self, X, input_dim):
        """Initialize InputLSTMLayer class

        Parameters
        ----------
        X : tf.Tensor
            Input tensor to be transformed
        input_dim : integer
            Input dimensions
        """
        self.X = X
        self.input_dim = input_dim

    def get_outputs(self):
        """Generate outputs for InputLSTMLayer class

        Returns
        ----------
        tf.Tensor
            Transformed input tensor
        """
        return tf.expand_dims(self.X, 0)
