import pytest
import timeflow as tflow


def output_lstm_layer_test():
    input_dim = 10
    output_dim = 10
    test_placeholder = tflow.placeholders.prediction.input_placeholder(input_dim)
    in_layer = tflow.layers.InputLSTMLayer(test_placeholder, input_dim, batch_input=False)
    out_layer = tflow.layers.OutputLSTMLayer(output_dim, in_layer, batch_output=False)
    out_layer_batch = tflow.layers.OutputLSTMLayer(output_dim, in_layer, batch_output=True)
    return out_layer.get_outputs(), out_layer_batch.get_outputs()


def test_output_lstm_layer():
    out_layer_out, out_layer_batch_out = output_lstm_layer_test()
    assert len(out_layer_batch_out.get_shape()) == 3
    assert len(out_layer_out.get_shape()) == 2
