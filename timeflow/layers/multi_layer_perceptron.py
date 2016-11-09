from .nn_layer import NNLayer
import tensorflow as tf
__all__ = ['MultiNNLayer']


class MultiNNLayer(NNLayer):
    """
    Layer implements the Multiple layer neural network.
    """
    def __init__(self, input_size, output_size, input_layer, layers=2,
                 layer_size=[10, 10], func='sigmoid', outfunc='regression'):
        """Initialize MultiNNLayer class

        Parameters
        ----------
        input_size : integer
            Input dimensions
        output_size : integer
            Output dimensions
        input_layer : layers object
            Preceding layers object
        layers : integer (default 2)
            Number of layers
        layer_size : list (default [10, 10])
            Size of layers
        func : string (default 'sigmoid')
            Layer function. Available choices are 'sigmoid', 'tanh' and 'relu'
        outfunc : string (default 'regression')
            Output type. Available choices are 'regression' and 'classification'

        """
        if layers != len(layer_size):
            raise ValueError("Layer information incorrect")
        self.inputs = input_layer.get_outputs()
        self.layers = layers
        self.layer_size = layer_size
        self.func = {'sigmoid': tf.nn.sigmoid,
                     'tanh': tf.nn.tanh,
                     'relu': tf.nn.relu}[func]
        self.outfunc = outfunc
        self.weights = {}
        self.biases = {}
        self.next_input = input_size
        self.next_output = layer_size[0]
        for i in range(layers):
            self.weights['h'+str(i+1)] = tf.Variable(tf.truncated_normal([self.next_input,
                                                                        self.next_output],mean=0,stddev=.01))
            self.biases['b'+str(i+1)] = tf.Variable(tf.truncated_normal([self.next_output],mean=0,stddev=.01))
            self.next_input = layer_size[i]
            if i+1 >= layers:
                self.next_output = output_size
            else:
                self.next_output = layer_size[i+1]
        self.weights['out'] = tf.Variable(tf.truncated_normal([self.next_input,
                                                                        self.next_output],mean=0,stddev=.01))
        self.biases['out'] = tf.Variable(tf.truncated_normal([self.next_output],mean=0,stddev=.01))

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
        variables = {}
        for i in range(self.layers):
            if i == 0:
                variables[i] = tf.add(tf.matmul(input_, self.weights['h'+str(i+1)]), self.biases['b'+str(i+1)])
                variables[i] = self.func(variables[i])
            else:
                variables[i] = tf.add(tf.matmul(variables[i-1], self.weights['h'+str(i+1)]), self.biases['b'+str(i+1)])
                variables[i] = self.func(variables[i])
        if self.outfunc == 'regression':
            output = tf.matmul(variables[self.layers-1], self.weights['out']) + self.biases['out']
        elif self.outfunc == 'classification':
            output = tf.nn.softmax(tf.matmul(variables[self.layers-1], self.weights['out']) + self.biases['out'])
        else:
            raise ValueError("Wrong output function provided")
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
