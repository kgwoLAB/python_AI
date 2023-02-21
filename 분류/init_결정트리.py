# 결정트리 

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

iris = load_iris()
X = iris.data[:,2:]
y = iris.target

tree_clf = DecisionTreeClassifier(max_depth=2)
tree_clf.fit(X,y)

from sklearn.tree import export_graphviz

export_graphviz(tree_clf,
                out_file="iris_tree.dot",
                feature_names = iris.feature_names[2:],
                class_names=iris.target_names,
                rounded=True,
                filled=True)


# 제한없이 결정트리를 훈련시키고 불필요한 노드를 가지치기 하는 알고리즘이 있다. 

#$ dot -Tpng iris_tree.dot -o iris_tree.png

# 회귀
from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor(max_depth=2)
tree_reg.fit(X,y)






from sklearn.datasets import make_moons
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

from sklearn.model_selection import GridSearchCV

X, y = make_moons(n_samples=1000, noise=0.4)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

from sklearn.preprocessing import StandardScaler


ss = StandardScaler()

X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

params = {'max_leaf_nodes': list(range(2, 100)), 'min_samples_split': [2, 3, 4]}
Grid_DT = GridSearchCV(DecisionTreeClassifier(random_state=42), params, verbose=1, cv=3)
Grid_DT_fit = Grid_DT.fit(X_train,y_train)
Grid_DT_fit.best_params_
Grid_DT_pred = Grid_DT_fit.best_estimator_.predict(X_train)

from sklearn.metrics import accuracy_score
accuracy_score(y_train,Grid_DT_pred)

mini_sets = []
from sklearn.model_selection import ShuffleSplit
rs = ShuffleSplit(n_splits=1000, test_size=len(X_train)-100,random_state=42)
for mini_train_index, mini_test_index in rs.split(X_train):
    X_mini_train = X_train[mini_train_index]
    y_mini_train = y_train[mini_train_index]
    mini_sets.append((X_mini_train, y_mini_train))

import numpy as np 
from sklearn.base import clone

forest = [clone(Grid_DT_fit.best_estimator_) for _ in range(1000)]

accuracy_scores = []

for tree, (X_mini_train, y_mini_train) in zip(forest, mini_sets):
    tree.fit(X_mini_train, y_mini_train)
    
    y_pred = tree.predict(X_test)
    accuracy_scores.append(accuracy_score(y_test, y_pred))

np.mean(accuracy_scores)


Y_pred = np.empty([1000, len(X_test)], dtype=np.uint8)
for tree_index, tree in enumerate(forest):
    Y_pred[tree_index] = tree.predict(X_test)
    
    
from scipy.stats import mode

y_pred_majority_votes, n_votes = mode(Y_pred, axis=0)
accuracy_score(y_test, y_pred_majority_votes.reshape([-1]))