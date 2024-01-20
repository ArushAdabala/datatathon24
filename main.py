import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn as skl
from sklearn.linear_model import ElasticNet
from sklearn import preprocessing
import math
from cleaning import clean_data

# I hate being responsible
pd.options.mode.chained_assignment = None  # default='warn'

# Plot coordinate values
# plt.plot(df[['surface_x', 'bh_x']].T, df[['surface_y', 'bh_y']].T, 'r')
# plt.scatter(df['surface_x'], df['surface_y'], c=np.log(df['OilPeakRate'] + np.e))
# plt.show()
df = clean_data()

df_arr = np.float64(df.to_numpy())

robust = preprocessing.StandardScaler()

training = df_arr[:-1,:-1]
robust_training = robust.fit_transform(training)
# Solve directly with lstsq
# https://stackoverflow.com/questions/21827594/raise-linalgerrorsvd-did-not-converge-linalgerror-svd-did-not-converge-in-m
x, residuals, rank, s = np.linalg.lstsq(robust_training, df_arr[:-1,-1], rcond=None)

testing = [df_arr[-1,:-1]]
robust_testing = robust.transform(testing)
print("Testing: prediction vs actual", robust_testing @ x, df_arr[-1,-1])

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
print(residuals.shape)
plt.plot(range(len(residuals)), residuals, 'r.')
plt.title("Residuals")
plt.show()