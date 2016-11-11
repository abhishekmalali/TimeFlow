import numpy as np


# Function to generate the irregular time prediction features
def irregular_prediction(time_samples, samples):
    """Initialize input placeholder for prediction networks

    Parameters
    ----------
    time_samples : np.array
        Time samples
    samples : np.array
        Signal samples

    Returns
    ----------
    np.array
        Data with features
    np.array
        Response
    np.array
        Time vector

    """
    delta_t = [0] + list(np.array(time_samples[1:]) - np.array(time_samples[:-1]))
    derivative = np.divide(np.diff(samples), np.diff(time_samples))
    first_derivative = np.lib.pad(derivative, (1, 0), 'constant',
                                  constant_values=(0, 0))
    second_derivative = np.lib.pad(np.diff(first_derivative), (1, 0),
                                   'constant', constant_values=(0, 0))
    data_mat = np.matrix([list(samples), delta_t, list(first_derivative),
                         list(second_derivative)]).T
    data_mat = data_mat[:-1, :]
    resp_mat = np.matrix(np.roll(samples, -1)[:-1]).T
    time_vec = time_samples[1:]
    return data_mat, resp_mat, time_vec
