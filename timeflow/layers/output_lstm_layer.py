from .nn_layer import NNLayer
import tensorflow as tf
__all__ = ['OutputLSTMLayer']


class OutputLSTMLayer(NNLayer):
    """
    This layer squashes dimensions for the LSTM output to be usable.
    The input data has a shape of (1, ?, output_dimensions) and the output
    for this layer is a tensor of the shape (?, output_dimensions)
    """
    def __init__(self, output_dim, input_layer):
        """Initialize InputLSTMLayer class

        Parameters
        ----------
        output_dim : integer
            Output dimensions
        input_layer : layers object
            Preceding layers object
        """
        self.output_dim = output_dim
        self.outputs = input_layer.get_outputs()

    def get_outputs(self):
        """Generate outputs for InputLSTMLayer class

        Returns
        ----------
        tf.Tensor
            Transformed output tensor
        """
        return self.outputs[0, :, :]
