import pytest
import timeflow as tflow


def mulitnnlayer_test():
    input_dim = 10
    output_dim = 10
    test_placeholder = tflow.placeholders.prediction.input_placeholder(input_dim)
    in_layer = tflow.layers.InputLSTMLayer(test_placeholder, input_dim, batch_input=False)
    multinnlayer = tflow.layers.MultiNNLayer(input_dim, output_dim,
                                             in_layer)
    multinnlayer_cls = tflow.layers.MultiNNLayer(input_dim, output_dim,
                                                 in_layer,
                                                 outfunc='classification')
    return multinnlayer.get_outputs(), multinnlayer_cls.get_outputs()


def test_multinnlayer():
    regout, clsout = mulitnnlayer_test()
    assert int(regout.get_shape()[2]) == 10
    assert int(clsout.get_shape()[2]) == 10
