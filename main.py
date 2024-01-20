import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn as skl
from sklearn.linear_model import ElasticNet
import math
from cleaning import clean_data

# I hate being responsible
pd.options.mode.chained_assignment = None  # default='warn'

# Plot coordinate values
# plt.plot(df[['surface_x', 'bh_x']].T, df[['surface_y', 'bh_y']].T, 'r')
# plt.scatter(df['surface_x'], df['surface_y'], c=np.log(df['OilPeakRate'] + np.e))
# plt.show()
df = clean_data()
for column in df:
    df = df[~df[column].isin([math.inf])]

df_arr = np.float64(df.to_numpy())


# Solve directly with lstsq
# https://stackoverflow.com/questions/21827594/raise-linalgerrorsvd-did-not-converge-linalgerror-svd-did-not-converge-in-m
x, residuals, rank, s = np.linalg.lstsq(df_arr[:-1,:-1], df_arr[:-1,-1], rcond=None)

print("Testing: prediction vs actual", df_arr[-1,:-1] @ x, df_arr[-1,-1])

# Solve using ElasticNet to get more reasonable coefficients
# model_alpha = 15  # 10 for images
# l1_ratio = 0.8  # 1.0 is LASSO, 0.1 is close to ridge
# model = ElasticNet(alpha=model_alpha, l1_ratio=l1_ratio, fit_intercept=False, selection='random', max_iter=10000)
# model.fit(material_arr, target_vector)
# x = model.coef_

# Set tiny coefficients to zero and print the number of nonzero coefficients
x[abs(x) < 1e-10] = 0
print(f"Number of nonzero coefficients: {np.argwhere(x).shape[0]}")
print(f"Percent of coefficients that are nonzero: {100 * np.argwhere(x).shape[0] / x.shape[0]}%")

# Plot coefficients
plt.plot(x, label=list(df.head()))
plt.show()
