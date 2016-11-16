import pytest
import timeflow as tflow
import numpy as np


def irregular_prediction_test():
    size = 100
    time_samples = sorted(np.random.normal(size=size))
    samples = np.random.normal(size=size)
    features, response, time_vector = \
        tflow.features.irregular_prediction(time_samples, samples)
    return features, response, time_vector


def test_irregular_prediction():
    features, response, time_vector = irregular_prediction_test()
    assert features.shape == (99, 4)
    assert response.shape == (99, 1)
    assert len(time_vector) == 99
