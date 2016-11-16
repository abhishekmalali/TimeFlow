import pytest
import timeflow as tflow


def lstm_layer_test():
    input_dim = 10
    batch_size = 10
    hidden_layer_size = 10
    test_placeholder = tflow.placeholders.prediction.input_placeholder(input_dim)
    batch_test_placeholder = tflow.placeholders.prediction.input_batch_placeholder(input_dim, batch_size)
    in_layer = tflow.layers.InputLSTMLayer(test_placeholder, input_dim, batch_input=False)
    in_layer_batch = tflow.layers.InputLSTMLayer(batch_test_placeholder, input_dim, batch_input=True)
    lstm_layer = tflow.layers.LSTMLayer(input_dim, hidden_layer_size, in_layer)
    lstm_layer_batch = tflow.layers.LSTMLayer(input_dim, hidden_layer_size, in_layer_batch)
    return lstm_layer.get_outputs(), lstm_layer_batch.get_outputs(), hidden_layer_size


def test_lstm_layer():
    lstm_layer_out, lstm_layer_batch_out, hsize = lstm_layer_test()
    shape = lstm_layer_out.get_shape()
    assert len(lstm_layer_out.get_shape()) == 3
    assert int(shape[2]) == hsize
    shape = lstm_layer_batch_out.get_shape()
    assert len(lstm_layer_batch_out.get_shape()) == 3
    assert int(shape[2]) == hsize
