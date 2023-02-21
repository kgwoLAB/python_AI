# 대충 적힌거 정리
# 1. 배치 경사 하강법
"""_summary_
# 경사 하강법
# 배치 경사 하강법
# 확률적 경사 하각법
# 미니배치 경사 하강법

# 릿소, 라쏘, 엘라스틱넷
"""

# 꽃 품종
from sklearn import datasets
iris = datasets.load_iris()
list(iris.keys())
x = iris["data"][:,3:]
y = (iris["target"] ==2).astype(int)

from sklearn.linear_model import LogisticRegression

log_reg =LogisticRegression()
log_reg.fit(x,y)

# 꽃 너비가 0~3cm 인 꽃에 대해 모델의 추정 확률 계산
import numpy as np
X_new = np.linspace(0,3,1000).reshape(-1,1)
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression(solver="lbfgs", random_state=42)
log_reg.fit(x, y)


y_proba = log_reg.predict(X_new)
y_proba.shape
import matplotlib.pyplot as plt
plt.plot(X_new, y_proba[:,1], "g--", label="Iris virginica")



X = iris["data"][:, 3:]  # 꽃잎 너비
y = (iris["target"] == 2).astype(int)  # Iris virginica이면 1 아니면 0


from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression(solver="lbfgs", random_state=42)
log_reg.fit(X, y)
X_new = np.linspace(0, 3, 1000).reshape(-1, 1)
y_proba = log_reg.predict_proba(X_new)
plt.plot(X_new, y_proba[:, 1], "g-", linewidth=2, label="Iris virginica")
plt.plot(X_new, y_proba[:, 0], "b--", linewidth=2, label="Not Iris virginica")


X_new = np.linspace(0, 3, 1000).reshape(-1, 1)
y_proba = log_reg.predict_proba(X_new)
decision_boundary = X_new[y_proba[:, 1] >= 0.5][0]

plt.figure(figsize=(8, 3))
plt.plot(X[y==0], y[y==0], "bs")
plt.plot(X[y==1], y[y==1], "g^")
plt.plot([decision_boundary, decision_boundary], [-1, 2], "k:", linewidth=2)
plt.plot(X_new, y_proba[:, 1], "g-", linewidth=2, label="Iris virginica")
plt.plot(X_new, y_proba[:, 0], "b--", linewidth=2, label="Not Iris virginica")
plt.text(decision_boundary+0.02, 0.15, "Decision  boundary", fontsize=14, color="k", ha="center")
plt.arrow(decision_boundary[0], 0.08, -0.3, 0, head_width=0.05, head_length=0.1, fc='b', ec='b')
plt.arrow(decision_boundary[0], 0.92, 0.3, 0, head_width=0.05, head_length=0.1, fc='g', ec='g')
plt.xlabel("Petal width (cm)", fontsize=14)
plt.ylabel("Probability", fontsize=14)
plt.legend(loc="center left", fontsize=14)
plt.axis([0, 3, -0.02, 1.02])
save_fig("logistic_regression_plot")
plt.show()