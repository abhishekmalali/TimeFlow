import pytest
import timeflow as tflow


def input_lstm_layer_test():
    input_dim = 10
    batch_size = 10
    test_placeholder = tflow.placeholders.prediction.input_placeholder(input_dim)
    batch_test_placeholder = tflow.placeholders.prediction.input_batch_placeholder(input_dim, batch_size)
    in_layer = tflow.layers.InputLSTMLayer(test_placeholder, input_dim, batch_input=False)
    in_layer_batch = tflow.layers.InputLSTMLayer(batch_test_placeholder, input_dim, batch_input=True)
    return in_layer.get_outputs(), in_layer_batch.get_outputs()


def test_input_lstm_layer():
    in_layer_out, in_layer_batch_out = input_lstm_layer_test()
    assert len(in_layer_out.get_shape()) == 3
    assert len(in_layer_batch_out.get_shape()) == 3
