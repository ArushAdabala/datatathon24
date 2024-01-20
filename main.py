import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn as skl
from sklearn.linear_model import ElasticNet
from sklearn import preprocessing
import math
from cleaning import clean_data
from compute import model

# I hate being responsible
# pd.options.mode.chained_assignment = None  # default='warn'
pd.set_option('chained_assignment',None)

# Plot coordinate values
# plt.plot(df[['surface_x', 'bh_x']].T, df[['surface_y', 'bh_y']].T, 'r')
# plt.scatter(df['surface_x'], df['surface_y'], c=np.log(df['OilPeakRate'] + np.e))
# plt.show()
df = clean_data()

df_arr = np.float64(df.to_numpy())

computer = model(df_arr, 100)
computer.scale()

# Solve directly with lstsq
x, residuals, rank, s = np.linalg.lstsq(computer.training_data, computer.training_results, rcond=None)

computer.set_results(x)

print("Linear Model Testing RMSE: ", computer.get_MSE())

# Plot some examples of predictions
computer.plot()



# Solve using ElasticNet to get more reasonable coefficients
# model_alpha = 15  # 10 for images
# l1_ratio = 0.8  # 1.0 is LASSO, 0.1 is close to ridge
# model = ElasticNet(alpha=model_alpha, l1_ratio=l1_ratio, fit_intercept=False, selection='random', max_iter=10000)
# model.fit(material_arr, target_vector)
# x = model.coef_

# Set tiny coefficients to zero and print the number of nonzero coefficients
x[abs(x) < 1e-5] = 0
print(f"Number of nonzero coefficients: {np.argwhere(x).shape[0]}")
print(f"Percent of coefficients that are nonzero: {100 * np.argwhere(x).shape[0] / x.shape[0]}%")

#Plot coefficients
plt.bar(list(df.head())[:-1], x)
plt.xticks(fontsize=10, rotation=-90)
plt.title("Coefficients")
plt.show()

# Plot residuals
# TODO: Residuals isn't the right thing to plot I think
# print(residuals.shape)
# plt.plot(range(len(residuals)), residuals, 'r.')
# plt.title("Residuals")
# plt.show()