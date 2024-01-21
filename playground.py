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

# Plot coordinate values
# plt.plot(df[['surface_x', 'bh_x']].T, df[['surface_y', 'bh_y']].T, 'r')
# plt.scatter(df['surface_x'], df['surface_y'], c=np.log(df['OilPeakRate'] + np.e))
# plt.show()
df = clean_data()
df = remove_correlations(df,0.9)

print(df)

