import pandas as pd

from sklearn.datasets import load_boston

from sklearn import datasets
X, y = datasets.fetch_openml('boston', return_X_y=True)

X.head()


X.info()


y.info()



import matplotlib.pyplot as plt
plt.scatter(X["RM"],y)
X.shape

Xx = X["RM"]
Xx.shape