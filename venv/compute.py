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
        self.training = robust.fit_transform(self.training_data)
        self.testing = robust.transform(self.testing_data)
    
    def set_results(self, results):
        self.results = self.testing_data @ results
    
    def get_MSE(self):
        mse = 0
        for i in range(self.num_tests):
            mse += (self.actual[i] - self.results[i]) ** 2

        return math.sqrt(mse / self.num_tests)
    
    def plot(self):
        x_axis = np.arange(self.num_tests)
        plt.bar(x_axis - 0.2, self.results, 0.4, 'prediction')
        plt.bar(x_axis + 0.2, self.actual, 0.4, 'actual')
        plt.title("Sample of predictions vs actual")
        plt.show()