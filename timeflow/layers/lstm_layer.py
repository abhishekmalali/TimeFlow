from .nn_layer import NNLayer
__all__ = ['LSTMLayer']


class LSTMLayer(NNLayer):
    """
    This layer implements the LSTM cell.
    """
    def __init__(self, input_dim, hidden_layer_size, input_layer):
        """Initialize LSTMLayer class

        Parameters
        ----------
        input_dim : integer
            Input dimensions
        hidden_layer_size : integer
            Size of the memory in LSTM cell
        input_layer : layers object
            Preceding layers object

        """
        self.input_dim = input_dim
        self.hidden_layer_size = hidden_layer_size
        self.inputs = input_layer.get_outputs()

        # Initializing the weights and biases
        self.Wi = tf.Variable(tf.zeros([self.input_dim, self.hidden_layer_size]))
        self.Ui = tf.Variable(tf.zeros([self.hidden_layer_size, self.hidden_layer_size]))
        self.bi = tf.Variable(tf.zeros([self.hidden_layer_size]))

        self.Wf = tf.Variable(tf.zeros([self.input_dim, self.hidden_layer_size]))
        self.Uf = tf.Variable(tf.zeros([self.hidden_layer_size, self.hidden_layer_size]))
        self.bf = tf.Variable(tf.zeros([self.hidden_layer_size]))

        self.Wog = tf.Variable(tf.zeros([self.input_dim, self.hidden_layer_size]))
        self.Uog = tf.Variable(tf.zeros([self.hidden_layer_size, self.hidden_layer_size]))
        self.bog = tf.Variable(tf.zeros([self.hidden_layer_size]))

        self.Wc = tf.Variable(tf.zeros([self.input_dim, self.hidden_layer_size]))
        self.Uc = tf.Variable(tf.zeros([self.hidden_layer_size, self.hidden_layer_size]))
        self.bc = tf.Variable(tf.zeros([self.hidden_layer_size]))

        self.initial_hidden = tf.zeros([1, self.hidden_layer_size])
        self.initial_hidden= tf.pack([self.initial_hidden, self.initial_hidden])

    def forward_step(self, previous_memory, input_):
        """
        Generates the next forward LSTM operation

        Parameters
        ----------
        previous_memory : list
            List of the previous memory and hidden output tensors
        input_ : tf.tensor
            Input tensor

        Returns
        ----------
        list
            New updated memory and hidden output tensors

        """
        previous_hidden_state, c_prev = tf.unpack(previous_memory)
        # Input gate
        i= tf.sigmoid(
            tf.matmul(input_,self.Wi)+tf.matmul(previous_hidden_state,self.Ui) + self.bi
        )
        # Forget Gate
        f= tf.sigmoid(
            tf.matmul(input_,self.Wf)+tf.matmul(previous_hidden_state,self.Uf) + self.bf
        )
        # Output Gate
        o= tf.sigmoid(
            tf.matmul(input_,self.Wog)+tf.matmul(previous_hidden_state,self.Uog) + self.bog
        )
        # New Memory Cell
        c_= tf.nn.tanh(
            tf.matmul(input_,self.Wc)+tf.matmul(previous_hidden_state,self.Uc) + self.bc
        )
        # Final Memory cell
        c= f*c_prev + i*c_

        # Current Hidden state
        current_hidden_state = o*tf.nn.tanh(c)
        return tf.pack([current_hidden_state,c])

    # Function for getting all hidden state.
    def get_outputs(self):
        """
        Iterates through time/ sequence to get all hidden state

        Returns
        ----------
        tf.Tensor
            Output tensor

        """
        # Getting all hidden state throuh time
        all_hidden_states = tf.scan(self.forward_step,
                                    self.inputs,
                                    initializer=self.initial_hidden,
                                    name='states')
        all_hidden_states = all_hidden_states[:, 0, :, :]
        return all_hidden_states
