import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn as skl

# using loadtxt()
arr = np.loadtxt("data/training.csv",
                 delimiter=",", dtype=str)

df = pd.read_csv("data/training.csv")
df = df.drop('Unnamed: 0', axis=1)
df.dropna(subset = ['bh_x','bh_y'], inplace = True)

for o in range(15):
    df.drop(df['OilPeakRate'].idxmax(), inplace=True)
    df.drop(df['OilPeakRate'].idxmin(), inplace=True)


# plt.plot(df[['surface_x', 'bh_x']].T, df[['surface_y', 'bh_y']].T, 'r')
plt.scatter(df['surface_x'], df['surface_y'], c=np.log(df['OilPeakRate'] + np.e))
plt.show()
