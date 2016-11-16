import pytest
import timeflow as tflow
import numpy as np


def plotting_test():
    size = 200
    num_training_points = 100
    num_validation_points = 50
    time_vector = np.array(sorted(np.random.normal(size=size)))
    predicted_series = np.random.normal(size=size)
    actual_series = np.random.normal(size=size)
    tflow.utils.plotting.plot_residuals(predicted_series,
                                        actual_series,
                                        time_vector,
                                        num_training_points,
                                        num_validation_points)


def test_plotting():
    plotting_test()
