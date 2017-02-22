import tensorflow as tf

class AutoCorrInitializer():
    """
    AutoCorrInitializer class does data munging on the outputs to bring them
    to the right dimensions to be fed into AutoCorr.
    """
    def __init__(self, inputs, outputs, y):
        self.inputs = inputs
        self.outputs = outputs
        self.y = y

    def create_tensor_output(self):
        # Separating out the time differences from the input vector
        time_diff = tf.expand_dims(self.inputs[:,1], axis=1)
        # Calculating the residuals
        residuals = tf.subtract(self.y, self.outputs)
        # Calculating the signal vector
        next_residual = residuals[1:]
        prev_residual = residuals[:-1]
        time_diff_residual = time_diff[1:]
        sign_res = tf.concat([next_residual, prev_residual], 1)
        return time_diff_residual, sign_res

class AutoCorr():
    """
    AutoCorr class estimates the autocorrelation coefficient from residuals.
    """
    def __init__(self, time_input, signal_input):
        """Initialize AutoCorr class

        Parameters
        ----------
        time_input : tf.Tensor
            Input vector/tensor with time difference values
        time_input : tf.Tensor
            Input vector/tensor with signal magnitude values
        """
        self.time_input = time_input
        self.signal_input = signal_input
        with tf.variable_scope('phi'):
            self.phi_tf = tf.Variable(tf.truncated_normal([1], mean=0.5, stddev=.01))
        with tf.variable_scope('sigma'):
            self.sigma_tf = tf.Variable(tf.truncated_normal([1], mean=0.5, stddev=.01),
                                        trainable=False)
        self.next_signal, self.prev_signal = tf.unstack(self.signal_input, axis=1)
        # Extending the dimensions of both the vectors
        self.next_signal = tf.expand_dims(self.next_signal, axis=1)
        self.prev_signal = tf.expand_dims(self.prev_signal, axis=1)
        # Packing all the three tensors for computation
        self.input_ = tf.stack([self.time_input, self.next_signal, self.prev_signal], axis=2)


    def generate_log_loss(self):
        """
        Returns
        ----------
        tf.Tensor
            Tensor with log-likelihood value
        """
        nu = tf.multiply(self.sigma_tf, tf.sqrt(tf.subtract(1., tf.pow(self.phi_tf, tf.multiply(2., self.time_input)))))
        e = tf.subtract(tf.multiply(tf.pow(self.phi_tf, self.time_input), self.prev_signal), self.next_signal)
        nu_sq = tf.pow(nu, 2.)
        e_sq = tf.pow(e, 2.)
        # Calculating all steps of LL
        log_lik = tf.log(tf.multiply(2.51, nu)) + tf.div(e_sq, tf.multiply(2., nu_sq))
        return tf.reduce_sum(log_lik)
