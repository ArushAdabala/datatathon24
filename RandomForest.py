import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn as skl
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn import utils
from cleaning import clean_data
from compute import model

pd.set_option('chained_assignment',None)
df = clean_data()
df_arr = np.float64(df.to_numpy())

computer = model(df_arr, 100)
label = preprocessing.LabelEncoder()

computer.scale()

rf_model = RandomForestClassifier()
print("hi")
print(label.fit_transform(computer.training_results))
#rf_model.fit(computer.training_data, label.fit_transform(computer.training_results))

#computer.set_results(rf_model.predict(computer.testing_data))

#print(computer.get_RMSE())



