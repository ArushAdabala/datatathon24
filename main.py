import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn as skl

# using loadtxt()
arr = np.loadtxt("data\\training.csv",
                 delimiter=",", dtype=str)

coords = np.float32(arr[1:,1:3])

print(coords)

plt.plot(coords[:,0], coords[:,1], '.')
plt.show()