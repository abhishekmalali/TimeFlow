import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib import gridspec
sns.set_style("whitegrid")


def plot_residuals(predicted_series,
                   actual_series,
                   time_vector,
                   num_training_points,
                   num_validation_points,
                   base_path=None,
                   file_name=None): # pragma: no cover
    """Plotting function for plotting the predicted series with the residuals.

    Parameters
    ----------
    predicted_series : np.array
        Predicted vector
    actual_series : np.array
        Actual response vector
    time_vector : np.array
        Time vector
    num_training_points : integer
        Number of points used for training
    num_validation_points : integer
        Number of points used for validation
    base_path: string (default None)
        Base path for saving the plot
    base_path: string (default None)
        Base path for saving the plot
    file_name: string (default None)
        File name for the plot

    """

    fig = plt.figure(figsize=(15, 15))
    residuals = actual_series - predicted_series
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
    axarr = [0, 0]
    axarr[0] = plt.subplot(gs[0])
    axarr[0].plot(time_vector, predicted_series,
                  marker='o', label='Predicted', markersize=5,
                  alpha=0.9, linestyle='-', linewidth=.5)
    axarr[0].plot(time_vector, actual_series, marker='o', label='Actual',
                  markersize=5, linestyle='--')
    xlim = plt.xlim()
    axarr[0].set_ylim((min(predicted_series)-1, max(predicted_series)+1))
    axarr[0].axvspan(xlim[0], time_vector[num_training_points], alpha=0.1,
                     label='Training set', color='r')
    axarr[0].axvspan(time_vector[num_training_points],
                     time_vector[num_training_points + num_validation_points],
                     alpha=0.1,
                     label='Validation set', color='b')
    axarr[0].legend()
    axarr[0].set_title('Predicted vs Actual plot', fontsize=24)
    axarr[0].set_xlabel('Time', fontsize=20)
    axarr[0].set_ylabel('Magnitude', fontsize=20)

    axarr[1] = plt.subplot(gs[1])
    axarr[1].scatter(time_vector.tolist(), residuals.tolist())
    xlim = plt.xlim()
    ax2_ylim = axarr[1].get_ylim()
    axarr[1].axvspan(xlim[0], time_vector[num_training_points], alpha=0.1,
                     label='Training set', color='r')
    axarr[1].axvspan(time_vector[num_training_points],
                     time_vector[num_training_points + num_validation_points],
                     alpha=0.1,
                     label='Validation set', color='b')
    axarr[1].legend()
    axarr[1].set_title('Residual plot', fontsize=24)
    axarr[1].set_xlabel('Time', fontsize=20)
    axarr[1].set_ylabel('Residuals', fontsize=20)
    axarr[1].set_xlim([min(time_vector), max(time_vector)])
    axarr[1].set_ylim(ax2_ylim[0]-1, ax2_ylim[1]+1)
    if (base_path is not None and file_name is not None):
        plt.savefig(base_path+file_name)
        plt.close()
