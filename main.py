import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn as skl
from sklearn.linear_model import ElasticNet
from sklearn import preprocessing
import math
from cleaning import clean_data

# I hate being responsible
# pd.options.mode.chained_assignment = None  # default='warn'
pd.set_option('chained_assignment',None)

# Plot coordinate values
# plt.plot(df[['surface_x', 'bh_x']].T, df[['surface_y', 'bh_y']].T, 'r')
# plt.scatter(df['surface_x'], df['surface_y'], c=np.log(df['OilPeakRate'] + np.e))
# plt.show()
df = clean_data()

df_arr = np.float64(df.to_numpy())

robust = preprocessing.RobustScaler()

num_test = 1000
training = df_arr[:-num_test,:-1]
robust_training = robust.fit_transform(training)
# Solve directly with lstsq
# https://stackoverflow.com/questions/21827594/raise-linalgerrorsvd-did-not-converge-linalgerror-svd-did-not-converge-in-m
# x, residuals, rank, s = np.linalg.lstsq(df_arr[:-num_test,:-1], df_arr[:-num_test,-1], rcond=None)
x, residuals, rank, s = np.linalg.lstsq(robust_training, df_arr[:-num_test,-1], rcond=None)

testing = df_arr[-num_test:,:-1]
robust_testing = robust.transform(testing)

mse = 0
test_results = robust_testing @ x
test_actual = df_arr[-num_test:,-1]
for i in range(num_test):
    mse += (test_actual[i] - test_results[i]) ** 2
#linar normalization = np.linalg.norm(test_results - test_actual)
print("Linear Model Testing RMSE: ", math.sqrt(mse / num_test))

# Plot some examples of predictions
x_axis = np.arange(num_test)
plt.bar(x_axis-0.2, robust_testing @ x, 0.4, 'prediction')
plt.bar(x_axis+0.2, df_arr[-num_test:,-1], 0.4, 'actual')
plt.title("Sample of predictions vs actual")
plt.show()



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