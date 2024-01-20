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

df_stages = df[~df["number_of_stages"].isna()]

df = df[df["number_of_stages"].isna()]
df = df.drop('number_of_stages', axis=1)
df = df.drop('average_stage_length', axis=1)
df = df.drop('average_proppant_per_stage', axis=1)
df = df.drop('average_frac_fluid_per_stage', axis=1)
for column in df:
    percent = ((25058 - df[column].isna().sum()) / 25058)
    print(column + " : " + str(percent))

plt.plot(df[['surface_x', 'bh_x']].T, df[['surface_y', 'bh_y']].T, 'r')
plt.plot(df['surface_x'], df['surface_y'], 'b.')
plt.show()