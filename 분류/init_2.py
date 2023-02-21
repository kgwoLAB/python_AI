from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
mnist.keys()

X, y = mnist["data"], mnist["target"]
X.shape


import matplotlib as mpl
import matplotlib.pyplot as plt

some_digit = X[0]
some_digit_image = some_digit.reshape(28,28)

plt.imshow(some_digit_image, cmap="binary")
plt.axis("off")
plt.show()

import numpy as np
y = y.astype(np.uint8)


# 구분하기에 앞서 교제에서는 그냥 나누는대 나는 섞어버리겠다.

# 섞고 싶은대 방법을 찾아보자
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

# 5인지 구분해보기
y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)

from sklearn.linear_model import SGDClassifier
sgd_clf =SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_5)
sgd_clf.predict([some_digit])


# 성능 측정하기
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone

skfolds = StratifiedKFold(n_splits=3, random_state=42, shuffle=True)

for train_index, test_index in skfolds.split(X_train, y_train_5):
    clone_clf = clone(sgd_clf)
    X_train_folds = X_train[train_index]
    y_train_folds = y_train_5[train_index]
    
    X_test_folds = X_train[test_index]
    y_test_folds = y_train_5[test_index]
    
    clone_clf.fit(X_train_folds, y_train_folds)
    y_pred = clone_clf.predict(X_test_folds)
    n_correct = sum(y_pred == y_test_folds)
    print(n_correct/ len(y_pred))
    
    
# 위는 수동 아래는 자동이다.
from sklearn.model_selection import cross_val_score
cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy")



# 오차행렬
from sklearn.model_selection import cross_val_predict
y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)

from sklearn.metrics import confusion_matrix
confusion_matrix(y_train_5, y_train_pred)
#첫행은 5아님 이미지(음성클래스[진짜음성],[거짓양성]5라고 잘못분류)
# 두번쨰 행은  5이미지에 대한것(양성클래스[거짓음성],[진짜양성]정확히5마고 분류함)


# 정밀도와 재현율
from sklearn.metrics import precision_score, recall_score
precision_score(y_train_5, y_train_pred)
recall_score(y_train_5, y_train_pred)

# 조화 평균
from sklearn.metrics import f1_score
f1_score(y_train_5, y_train_pred)


# 정밀도/재현율 트레[이드 오프
# # 결정함수.. 임계값보다 크면 양성 그렇지 않으면 음성

y_scores = sgd_clf.decision_function([some_digit])
y_scores
threshold =0

y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3, method="decision_function")

from sklearn.metrics import precision_recall_curve
precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)

def plot_precision_recall_vs_threshold(precisions,recalls,thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="정밀도")
    plt.plot(thresholds, recalls[:-1],"g-", label="재현율")
    plt.xlim([-40000,40000])
    plt.grid(True)


plot_precision_recall_vs_threshold(precisions,recalls,thresholds)

plt.plot(recalls,precisions, "b--")

# 정밀도 수정
threholds_90_precision = thresholds[np.argmax(precisions >=0.90)]
y_train_pred_90 = (y_scores >= threholds_90_precision)

precision_score(y_train_5, y_train_pred_90)
recall_score(y_train_5, y_train_pred_90)



# ROC로 봐보기
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)





# 다중 분류
# OvR 결정 점수가 가장 높은 클래스 선택
# OvO 각 숫자 조합마다 이진 분류기를 훈련하는 전략
# 서포트 벡터 머신 분류기
from sklearn.svm import SVC
svm_clf = SVC()
svm_clf.fit(X_train, y_train)
svm_clf.predict([some_digit])

some_digit_scores = svm_clf.decision_functionm([some_digit])
some_digit_scores
# 클래스별 확률점수가 나옵니다. 

np.argmax(some_digit_scores)
svm_clf.classes_


from sklearn.multiclass import OneVsRestClassifier
ovr_clf = OneVsRestClassifier(SVC())
ovr_clf.fit(X_train,y_train)
ovr_clf.predict([some_digit])
len(ovr_clf.estimators_)

