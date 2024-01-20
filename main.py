import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn as skl

# using loadtxt()
arr = np.loadtxt("data\\training.csv",
                 delimiter=",", dtype=str)

df = pd.read_csv("data\\training.csv")
df = df.drop('Unnamed: 0', axis=1)
df.dropna(subset = ['bh_x','bh_y'], inplace = True)
dfxy = df[['surface_x','surface_y']]




plt.plot(df['surface_x'], df['surface_y'], '.')
plt.show()