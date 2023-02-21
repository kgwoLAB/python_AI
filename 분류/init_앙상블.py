from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.datasets import make_moons
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

from sklearn.model_selection import GridSearchCV

X, y = make_moons(n_samples=1000, noise=0.4)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)


log_clf = LogisticRegression()
rnd_clf = RandomForestClassifier()
svm_clf = SVC(probability=True)

voting_clf = VotingClassifier(
    estimators=[('lr',log_clf),('rf',rnd_clf),('svc',svm_clf)],
    voting='soft')
voting_clf.fit(X_train, y_train)
from sklearn.metrics import accuracy_score
for clf in (log_clf,rnd_clf, svm_clf, voting_clf):
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    print(clf.__class__.__name__, accuracy_score(y_test,y_pred))
    
    
    
# 배깅과 페이팅
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
bag_clf = BaggingClassifier(
    DecisionTreeClassifier(), n_jobs=-1)

params = {"n_estimators":range(1,1000), "max_samples":range(1,500), "bootstrap":(True,False),}
GS = GridSearchCV(BaggingClassifier(DecisionTreeClassifier(),n_jobs=-1),params, verbose=1, cv=3)

GS.fit(X_train, y_train)
GS.best_params_
GS.best_score__

bag_clf.fit(X_train,y_train)
y_pred = bag_clf.predict(X_train)
accuracy_score(y_train,y_pred)





# 연습문제
import numpy as np
from sklearn.datasets import fetch_openml
mnist = fetch_openml("mnist_784", version=1, cache=True)
X = mnist["data"]
y = mnist["target"].astype(np.uint8)

X_train = X[:50000]
y_train = y[:50000]
X_valid = X[50000:60000]
y_valid = y[50000:60000]
X_test  = X[60000:]
y_test  = y[60000:]

from sklearn.ensemble import ExtraTreesClassifier
end_clf = ExtraTreesClassifier()
rnd_clf = RandomForestClassifier()
svm_clf = SVC()
svm_clf = LogisticRegression()

voting_clf = VotingClassifier(
    estimators=[('lr',log_clf),('rf',rnd_clf),('svc',svm_clf),('end',end_clf)],
    voting='soft')


from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier

random_forest_clf = RandomForestClassifier(n_estimators=100, random_state=42)
extra_trees_clf = ExtraTreesClassifier(n_estimators=100, random_state=42)
svm_clf = LinearSVC(max_iter=100, tol=20, random_state=42)
mlp_clf = MLPClassifier(random_state=42)

estimators = [random_forest_clf, extra_trees_clf, svm_clf, mlp_clf]
for estimator in estimators:
    print("Training the", estimator)
    estimator.fit(X_train, y_train)
    
[estimator.score(X_val, y_val) for estimator in estimators]


유체역하ㅑㄱ