import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn as skl
from sklearn.linear_model import ElasticNet
from cleaning import clean_data

# I hate being responsible
pd.options.mode.chained_assignment = None  # default='warn'


# plt.plot(df[['surface_x', 'bh_x']].T, df[['surface_y', 'bh_y']].T, 'r')
# plt.scatter(df['surface_x'], df['surface_y'], c=np.log(df['OilPeakRate'] + np.e))
# plt.show()
df = clean_data()

df_arr = np.float64(df.to_numpy())
# Solve directly with lstsq
# https://stackoverflow.com/questions/21827594/raise-linalgerrorsvd-did-not-converge-linalgerror-svd-did-not-converge-in-m
x, residuals, rank, s = np.linalg.lstsq(df_arr[:-1,:-1], df_arr[:-1,-1], rcond=None)

print(df_arr[-1,:-1] @ x, df_arr[-1,-1])

# Solve using ElasticNet to get more reasonable coefficients
# model_alpha = 15  # 10 for images
# l1_ratio = 0.8  # 1.0 is LASSO, 0.1 is close to ridge
# model = ElasticNet(alpha=model_alpha, l1_ratio=l1_ratio, fit_intercept=False, selection='random', max_iter=10000)
# model.fit(material_arr, target_vector)
# x = model.coef_

# x[abs(x) < 1e-10] = 0
# print(np.argwhere(x).shape[0])

# Plot coefficients
# TODO: make it display the solumn names on the x axis
plt.plot(x)
plt.show()