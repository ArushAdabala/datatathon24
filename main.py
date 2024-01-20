import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn as skl
from sklearn.linear_model import ElasticNet

# I hate being responsible
pd.options.mode.chained_assignment = None  # default='warn'


def string_columns_to_float(df):
    # Convert all columns which are composed of strings to floats
    # input: df (dataframe)
    # output: none (modifies in-place)
    for colname in df.head():
        if any([isinstance(elem, str) for elem in df[colname]]):
            values = list(set(df[colname]))
            for i in range(len(values)):
                # https://stackoverflow.com/questions/20625582/how-to-deal-with-settingwithcopywarning-in-pandas
                df[colname][df[colname] == values[i]] = i+1


def print_corr(df):
    # Print columns in df that are closely correlated
    # Input: df (a dataframe)
    # Output: none (printed)
    corr = df.corr()
    # corr is a dataframe
    # Print highly correlated columns
    corr_arr = corr.to_numpy()
    for row in range(corr_arr.shape[0]):
        # Get all columns in upper triangle whose abs is greater than 0.5
        high_corr_idxs = np.nonzero(np.abs(corr_arr[row]) > 0.5)[0]
        # print(high_corr_idxs)
        for idx in [i for i in high_corr_idxs if i < row]:
            print(corr_arr[row, idx], list(corr.head())[row], list(corr.head())[idx])


# using loadtxt()
arr = np.loadtxt("data/training.csv",
                 delimiter=",", dtype=str)

df = pd.read_csv("data/training.csv")
df = df.drop('Unnamed: 0', axis=1)
##df.dropna(subset = ['bh_x','bh_y'], inplace = True)
df.dropna(subset = ['OilPeakRate'], inplace = True)

string_columns_to_float(df)

# for o in range(15):
#     df.drop(df['OilPeakRate'].idxmax(), inplace=True)
#     df.drop(df['OilPeakRate'].idxmin(), inplace=True)

df_stages = df[~df["number_of_stages"].isna()]
df = df[df["number_of_stages"].isna()]
df = df.drop('number_of_stages', axis=1)
df = df.drop('average_stage_length', axis=1)
df = df.drop('average_proppant_per_stage', axis=1)
df = df.drop('average_frac_fluid_per_stage', axis=1)
df = df.dropna()
# for column in df:
#     percent = ((25058 - df[column].isna().sum()) / 25058)
#     # print(column + " : " + str(percent))


# plt.plot(df[['surface_x', 'bh_x']].T, df[['surface_y', 'bh_y']].T, 'r')
# plt.scatter(df['surface_x'], df['surface_y'], c=np.log(df['OilPeakRate'] + np.e))
# plt.show()


df_arr = np.float64(df.to_numpy())

# ------Solve for coefficients-------
# Solve directly with lstsq
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