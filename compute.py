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
        x_axis = np.arange(num_bars)
        plt.bar(x_axis - 0.2, self.results[:num_bars], 0.4, 'prediction')
        plt.bar(x_axis + 0.2, self.actual[:num_bars], 0.4, 'actual')
        plt.title("Sample of predictions vs actual")
        plt.show()

    def plot_residuals(self, col_idx):
        # Plot residuals ordered by the value of column col_idx
        # Input: col_idx - index of column to sort data by
        print("Plotting residuals")
        diff = self.actual - self.results
        sort_idxs = np.argsort(self.testing_data[:, col_idx].T)
        plt.plot(range(len(diff)), diff[sort_idxs], 'r.')
        plt.title(f"Residuals ordered by column {col_idx}")
        plt.show()