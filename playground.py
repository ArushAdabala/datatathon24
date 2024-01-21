import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn as skl
from sklearn.linear_model import ElasticNet, ElasticNetCV
from cleaning import clean_data
from cleaning import remove_correlations
from cleaning import print_corr
from compute import model
pd.set_option('chained_assignment',None)


df = clean_data()
df, colnames = remove_correlations(df,0.9)
print(df)


# Plot coordinate values
plt.plot(df[['surface_x', 'bh_x']].T, df[['surface_y', 'bh_y']].T, 'r')
plt.scatter(df['surface_x'], df['surface_y'], c=np.log(df['OilPeakRate'] + np.e))
# plt.scatter(df['surface_x'], df['surface_y'], c=range(df.shape[0]))
# plt.colorbar()
plt.show()

# # Convert to array
# df_arr = np.float64(df.to_numpy())
# # Randomly shuffle rows so that there isn't any training/testing dataset bias
# np.random.shuffle(df_arr)
# Plot all positions of wells with indices to see that the shuffle was successful
# plt.scatter(df_arr[:,0], df_arr[:,1], c=range(df.shape[0]))
# plt.show()

print(df)

