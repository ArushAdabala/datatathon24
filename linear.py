import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn as skl
from sklearn.linear_model import ElasticNet, ElasticNetCV
from cleaning import clean_data
from compute import model

# I hate being responsible
# pd.options.mode.chained_assignment = None  # default='warn'
pd.set_option('chained_assignment',None)

df = clean_data()
# Plot coordinate values
plt.plot(df[['surface_x', 'bh_x']].T, df[['surface_y', 'bh_y']].T, 'r')
plt.scatter(df['surface_x'], df['surface_y'], c=np.log(df['OilPeakRate'] + np.e))
# plt.scatter(df['surface_x'], df['surface_y'], c=range(df.shape[0]))
plt.show()

# Load data into numpy array
df_arr = np.float64(df.to_numpy())
# Randomly shuffle rows so that there isn't any training/testing dataset bias
np.random.shuffle(df_arr)

# Plot all positions of wells with indices to see that the shuffle was successful
# plt.scatter(df_arr[:,0], df_arr[:,1], c=range(df.shape[0]))
# plt.show()

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
# computer.plot_bars(25)

# Plot the residuals
computer.plot_residuals(1)

# Plot coefficients
# plt.bar(list(df.head())[:-1], x)
# plt.xticks(fontsize=10, rotation=-90)
# plt.title("Coefficients")
# plt.show()

