import tensorflow as tf
__all__ = ['AutomatedTrainingMonitor']


class AutomatedTrainingMonitor:

    def __init__(self, input_var, output_var, training_input, training_output,
                 train, cost, sess, training_steps=100,
                 validation_input=None, validation_output=None,
                 early_stopping_rounds=None, burn_in=50):
        """Initialize AutomatedTrainingMonitor class

        Parameters
        ----------
        input_var : tf.tensor
            Input tensor
        output_var : tf.tensor
            Output tensor
        training_input : np.array
            Training input
        training_output : np.array
            Training output
        train : tf.Optimizer
            Optimizer for the network
        cost : tf.constant
            Cost/Loss function operation
        sess : tf.Session
            Tensorflow Session
        training_steps : integer (default 100)
            Training steps
        validation_input : np.array
            Validation input
        validation_output : np.array
            Validation output
        early_stopping_rounds : integer (default None)
            Number of iterations to check for early stopping
        burn_in : integer (default 50)
            Burn in period for the training
        """

        self.input_var = input_var
        self.output_var = output_var
        self.sess = sess
        self.cost = cost
        self.train_step = train
        self.training_input = training_input
        self.training_output = training_output
        self.training_steps = training_steps
        self.validation_input = validation_input
        self.validation_output = validation_output
        self.early_stopping_rounds = early_stopping_rounds
        self._best_value_step = None
        self._best_value = None
        self._early_stopped = False
        self.burn_in = burn_in

    @property
    def early_stopped(self):
        """Returns True if this monitor caused an early stop."""
        return self._early_stopped

    @property
    def best_step(self):
        """Returns the step at which the best early stopping metric was found."""
        return self._best_value_step

    @property
    def best_value(self):
        """Returns the best early stopping metric value found so far."""
        return self._best_value

    def validate_every_step(self, step):
        if self.early_stopping_rounds is not None:
            current_value = float(self.sess.run(self.cost,feed_dict={self.input_var:self.validation_input,
                                                                     self.output_var:self.validation_output}))
            if (self._best_value is None or current_value < self._best_value):
                self._best_value = current_value
                self._best_value_step = step
            stop_now = (step - self._best_value_step >= self.early_stopping_rounds)
            if stop_now:
                self._early_stopped = True
        return

    def train(self):
        for iter_num in range(self.training_steps):
            self.sess.run(self.train_step,feed_dict={self.input_var:self.training_input,
                                                     self.output_var:self.training_output})
            if iter_num >= self.burn_in:
                self.validate_every_step(iter_num)
            if self._early_stopped is True:
                break
        print "Final Validation loss: ",\
              float(self.sess.run(self.cost,feed_dict={self.input_var:self.validation_input,
                                                       self.output_var:self.validation_output}))
        print "Number of Iterations: ",\
              iter_num

    def reset_early_stopped(self):
        self._best_value_step = None
        self._best_value = None
        self._early_stopped = False
