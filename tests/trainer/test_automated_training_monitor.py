import pytest
import timeflow as tflow
import numpy as np
import tensorflow as tf


def run_automated_monitor():
    size = 300
    time_samples = sorted(np.random.normal(size=size))
    samples = np.random.normal(size=size)
    X, Y, time_vector = \
        tflow.features.irregular_prediction(time_samples, samples)
    num_training_points = 100
    num_validation_points = 100
    X_train = X[:num_training_points, :]
    Y_train = Y[:num_training_points, :]
    X_valid = X[num_training_points:num_training_points+num_validation_points, :]
    Y_valid = Y[num_training_points:num_training_points+num_validation_points, :]
    X_test = X[num_training_points+num_validation_points:, :]
    Y_test = Y[num_training_points+num_validation_points:, :]
    input_size = 4
    hidden_size = 10
    output_size = 1
    with tf.variable_scope('Input'):
        inputs = tflow.placeholders.prediction.input_placeholder(input_size)
    with tf.variable_scope('Input_LSTM_Layer'):
        input_lstm_layer = tflow.layers.InputLSTMLayer(inputs, input_size)
    with tf.variable_scope('LSTM_Layer'):
        lstm_layer = tflow.layers.LSTMLayer(input_size, hidden_size, input_lstm_layer)
    with tf.variable_scope('Regression_Layer'):
        reg_layer = tflow.layers.RegressionLayer(hidden_size, output_size, lstm_layer)
    with tf.variable_scope('Output_LSTM_Layer'):
        output_layer = tflow.layers.OutputLSTMLayer(output_size, reg_layer)
    y = tflow.placeholders.prediction.output_placeholder(output_size)
    outputs = output_layer.get_outputs()
    # Defining MSE as the loss function
    with tf.variable_scope('RMSE'):
        loss_func = tflow.utils.metrics.RMSE(outputs, y)
    train_step = tf.train.RMSPropOptimizer(learning_rate=0.05).minimize(loss_func)
    sess = tf.InteractiveSession()
    sess.run(tf.initialize_all_variables())
    monitor = tflow.trainer.AutomatedTrainingMonitor(inputs, y, X_train, Y_train,
                                                 train_step, loss_func, sess, training_steps=500,
                                                 validation_input=X_valid, validation_output=Y_valid,
                                                 early_stopping_rounds=60)
    monitor.train()
    return


def test_automated_training_monitor():
    run_automated_monitor()
