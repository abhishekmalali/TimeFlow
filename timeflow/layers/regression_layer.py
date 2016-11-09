from .nn_layer import NNLayer
import tensorflow as tf
__all__ = ['RegressionLayer']


class RegressionLayer(NNLayer):
    """
    Layer implements the Simple regression.
    """
    def __init__(self, input_size, output_size, input_layer):
        """Initialize RegressionLayer class

        Parameters
        ----------
        input_size : integer
            Input dimensions
        output_size : integer
            Output dimensions
        input_layer : layers object
            Preceding layers object

        """
        self.inputs = input_layer.get_outputs()
        self.input_size = input_size
        self.output_size = output_size
        self.Wo = tf.Variable(tf.truncated_normal([self.input_size, self.output_size], mean=0, stddev=.01))
        self.bo = tf.Variable(tf.truncated_normal([self.output_size], mean=0, stddev=.01))

    def get_output(self, input_):
        """
        Generates the output for a single step

        Parameters
        ----------
        input_ : tf.tensor
            Input tensor

        Returns
        ----------
        tf.tensor
            Output tensor

        """
        output = tf.matmul(input_, self.Wo) + self.bo
        return output

    def get_outputs(self):
        """
        Iterates through all inputs to generate outputs

        Returns
        ----------
        tf.Tensor
            Output tensor

        """
        all_outputs = tf.map_fn(self.get_output, self.inputs)
        return all_outputs
