import pytest
import timeflow as tflow


def regression_layer_test():
    input_dim = 10
    output_dim = 20
    test_placeholder = tflow.placeholders.prediction.input_placeholder(input_dim)
    in_layer = tflow.layers.InputLSTMLayer(test_placeholder, input_dim, batch_input=False)
    reg_layer = tflow.layers.RegressionLayer(input_dim, output_dim, in_layer)
    return reg_layer.get_outputs()


def test_regression_layer():
    regout = regression_layer_test()
    assert int(regout.get_shape()[2]) == 20
    assert len(regout.get_shape()) == 3
