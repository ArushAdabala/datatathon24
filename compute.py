import numpy
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn as skl
from sklearn import preprocessing
import math


class model():
    def __init__(self, data, num_tests):
        self.num_tests = num_tests
        self.training_data = data[:-num_tests, :-1]
        self.training_results = data[:-num_tests, -1]
        self.testing_data = data[-num_tests:, :-1]
        self.results = None
        self.actual = data[-num_tests:,-1]

    def scale(self):
        robust = preprocessing.RobustScaler()
        self.training_data = robust.fit_transform(self.training_data)
        self.testing_data = robust.transform(self.testing_data)
    
    def set_results(self, results):
        self.results = results
    
    def get_RMSE(self):
        # Preduce RMSE between self.actual and self.results
        diff = self.actual - self.results
        return np.sqrt(np.dot(diff, diff) / self.num_tests)
    
    def plot_bars(self, num_bars):
        # Plots a sample of predictions and their corresponding actual values
        # Input: num_bars is number of bars on chart
        print("Plotting bars")
        # x_axis = np.arange(num_bars)
        # plt.bar(x_axis - 0.2, self.results[:num_bars], 0.4, label='prediction')
        # plt.bar(x_axis + 0.2, self.actual[:num_bars], 0.4, label='actual')
        # plt.legend()
        # plt.title("Sample of predictions vs actual")
        # plt.show()
        annotated_bar_chart(num_bars, self.actual[:num_bars], self.results[:num_bars], list(range(num_bars)))

    def plot_residuals(self, col_idx):
        # Plot residuals ordered by the value of column col_idx
        # Input: col_idx - index of column to sort data by
        print("Plotting residuals")
        diff = self.actual - self.results
        sort_idxs = np.argsort(self.testing_data[:, col_idx].T)
        plt.plot(range(len(diff)), diff[sort_idxs], 'r.')
        plt.title(f"Residuals ordered by column {col_idx}")
        plt.show()


def annotated_bar_chart(sample_size, sampled_actuals, sampled_predictions, indices):
    # Create a bar plot showing the predicted vs actual values
    x = np.arange(sample_size)  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots(figsize=(14, 8))
    rects1 = ax.bar(x - width / 2, np.floor(sampled_actuals), width, label='Actual')
    rects2 = ax.bar(x + width / 2, np.floor(sampled_predictions), width, label='Predicted')

    # Add some text for labels, title, and custom x-axis tick labels, etc.
    ax.set_ylabel('Values')
    ax.set_title('Comparison of Actual and Predicted Values for Random Test Samples')
    ax.set_xticks(x)
    ax.set_xticklabels(indices)
    ax.legend()

    # Function to attach a text label above each bar in *rects*, displaying its height.
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()

    plt.show()


def well_prediction_comparison_plot(data, prediction, actual):
    # Plot the wells with predictions and actual side-by-side
    # Inputs:
    #  - data: an array with as many rows as prediciton and actual and whose first two columns are x and y
    #  - prediction: model's predictions of each well
    #  - actual: actual output of each well
    # Output: none (shows plots)
    ultimate_min = np.min(np.vstack((prediction, actual)))
    ultimate_max = np.max(np.vstack((prediction, actual)))
    plt.subplot(1, 2, 1)
    plt.title("Predictions")
    plt.scatter(data[:, 0], data[:, 1],
                c=np.log(prediction + np.e),
                vmin=np.log(ultimate_min + np.e), vmax=np.log(ultimate_max + np.e))

    plt.subplot(1, 2, 2)
    plt.title("Actual")
    plt.scatter(data[:, 0], data[:, 1],
                c=np.log(actual + np.e),
                vmin=np.log(ultimate_min + np.e), vmax=np.log(ultimate_max + np.e))

    plt.colorbar()
    plt.suptitle("Model vs Actual Comparison (log color scale)")
    plt.show()