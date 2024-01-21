import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn as skl
from sklearn.linear_model import ElasticNet, ElasticNetCV
from cleaning import clean_data, remove_correlations
from compute import model, well_prediction_comparison_plot

# Disable warnings (I hate being responsible)
# pd.options.mode.chained_assignment = None  # default='warn'
pd.set_option('chained_assignment',None)

df = clean_data()
df, colnames = remove_correlations(df, 0.8)
# Load data into numpy array
df_arr = np.float64(df.to_numpy())
# Randomly shuffle rows so that there isn't any training/testing dataset bias
np.random.shuffle(df_arr)


computer = model(df_arr, 1000)
computer.scale()

# Solve directly with lstsq
# x, residuals, rank, s = np.linalg.lstsq(computer.training_data, computer.training_results, rcond=None)

# Solve using ElasticNet to get more reasonable coefficients
# model_alpha = 1  # 10 for images
# l1_ratio = 0.4  # 1.0 is LASSO, 0.1 is close to ridge
# model = ElasticNet(alpha=model_alpha, l1_ratio=l1_ratio, fit_intercept=False, selection='random', max_iter=10000)
# Apparently ElasticNetCV is supposed to calculate its own parameters
model = ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, .99, 1], max_iter=10000)
model.fit(computer.training_data, computer.training_results)
x = model.coef_

# computer.set_results(x)
computer.set_results(model.predict(computer.testing_data))

print("Linear Model Testing RMSE: ", computer.get_RMSE())

# Plot some examples of predictions
computer.plot_bars(25)

# Plot the residuals
computer.plot_residuals(1)

# Plot coefficients
plt.bar(colnames[:-1], x)
plt.xticks(fontsize=10, rotation=-90)
plt.title("Coefficients of Linear Model")
plt.subplots_adjust(bottom=0.5)
plt.show()

# Plot the wells with predictions
well_prediction_comparison_plot(computer.testing_data, computer.results, computer.actual)