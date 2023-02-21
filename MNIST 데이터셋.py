from tensorflow import keras
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml

mnist = fetch_openml('mnist_784', version=1, as_frame=False)
#mnist.keys()

X, y = mnist["data"], mnist["target"]
#X.shape

y = y.astype(np.uint8)

X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]


some_X = X[:10]
some_y = y[:10]



import sklearn

from sklearn.neighbors import KNeighborsClassifier

knc = KNeighborsClassifier()
knc.fit(X_train,y_train)
knc.predict(some_X)

from sklearn.model_selection import cross_val_predict
y_train_knn_pred = cross_val_predict(knc, X_train, y_train, cv= 3)
from sklearn.metrics import f1_score
f1_score(y_train, y_train_knn_pred, average="macro")


from sklearn.model_selection import GridSearchCV
param = [
    {'weights':range(1,100)},
    {'n_neighbors':range(1,100)},
]
gs = GridSearchCV(knc, param, cv=5, scoring="accuracy",return_train_score=True, n_jobs=-1)
gs.fit(X_test, y_test)
gs.best_params_

# 타이타닉

train = pd.read_csv('kaggle/titanic/train.csv')
test = pd.read_csv('kaggle/titanic/test.csv')


import os
import urllib.request

TITANIC_PATH = os.path.join("datasets", "titanic")
DOWNLOAD_URL = "https://raw.githubusercontent.com/rickiepark/handson-ml2/master/datasets/titanic/"

def fetch_titanic_data(url=DOWNLOAD_URL, path=TITANIC_PATH):
    if not os.path.isdir(path):
        os.makedirs(path)
    for filename in ("train.csv", "test.csv"):
        filepath = os.path.join(path, filename)
        if not os.path.isfile(filepath):
            print("Downloading", filename)
            urllib.request.urlretrieve(url + filename, filepath)

fetch_titanic_data()   

import pandas as pd

def load_titanic_data(filename, titanic_path=TITANIC_PATH):
    csv_path = os.path.join(titanic_path, filename)
    return pd.read_csv(csv_path)


train_data = load_titanic_data("train.csv")
test_data = load_titanic_data("test.csv")

train_data.head()
train_data.info()
train_data.describe()

train_data["Survived"].value_counts()
train_data["Pclass"].value_counts()
train_data["Sex"].value_counts()
train_data["Embarked"].value_counts()

#Embarked 특성은 승객이 탑승한 곳을 알려 줍니다: C=Cherbourg, Q=Queenstown, S=Southampton.
 
train_data = train_data.set_index("PassengerId")
test_data = test_data.set_index("PassengerId")

train_data[train_data["Sex"]=="female"]["Age"].median()

from sklearn.impute import SimpleImputer

si = SimpleImputer(strategy="median")
si.fit(train_data)




from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

first_pl = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",StandardScaler()),
])


# 범주형 분류기
from sklearn.preprocessing import OneHotEncoder
second_pl = Pipeline([
    ("imputer",SimpleImputer(strategy="most_frequent")),
    ("cat_encoder",OneHotEncoder(sparse=False))
])

train_data.info()
no_objects = ["Age", "SibSp", "Parch", "Fare"]
objects = ["Pclass", "Sex", "Embarked"]


from sklearn.compose import ColumnTransformer
last_pl = ColumnTransformer([
    ("first", first_pl, no_objects),
    ("second", second_pl, objects),
    
])

X_train = last_pl.fit_transform(train_data[no_objects+objects])
X_train



from sklearn.preprocessing import PolynomialFeatures

X_train.shape
test = PolynomialFeatures(degree=2, include_bias=False)
x_test = test.fit_transform(X_train)
train_data.info()
test_arr = np.array(no_objects+objects).reshape(1,-1)
test_label = test.fit_transform(test_arr)

test_matrix = pd.DataFrame(x_test, columns=).corr()
test_matrix["Age"].sort_calvues(ascending=False)
plt.plot(x_test)
plt.plot(X_train)


y_train = train_data["Survived"]



from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=100, random_state=42)
rfc.fit(X_train,y_train)

X_test = last_pl.transform(test_data[no_objects + objects])
y_pred = rfc.predict(X_test)

from sklearn.model_selection import cross_val_score

forest_scores = cross_val_score(rfc, X_train, y_train, cv=10)
forest_scores.mean()

from sklearn.svm import SVC
svm_clf = SVC(gamma="auto")
svm_scores = cross_val_score(svm_clf, X_train,y_train, cv=10)
svm_scores.mean()


# 4. 스팸분류기
import os
import tarfile
import urllib.request

DOWNLOAD_ROOT = "http://spamassassin.apache.org/old/publiccorpus/"
HAM_URL = DOWNLOAD_ROOT + "20030228_easy_ham.tar.bz2"
SPAM_URL = DOWNLOAD_ROOT + "20030228_spam.tar.bz2"
SPAM_PATH = os.path.join("datasets", "spam")

def fetch_spam_data(ham_url=HAM_URL, spam_url=SPAM_URL, spam_path=SPAM_PATH):
    if not os.path.isdir(spam_path):
        os.makedirs(spam_path)
    for filename, url in (("ham.tar.bz2", ham_url), ("spam.tar.bz2", spam_url)):
        path = os.path.join(spam_path, filename)
        if not os.path.isfile(path):
            urllib.request.urlretrieve(url, path)
        tar_bz2_file = tarfile.open(path)
        tar_bz2_file.extractall(path=spam_path)
        tar_bz2_file.close()
        
        
fetch_spam_data()

HAM_DIR = os.path.join(SPAM_PATH, "easy_ham")
SPAM_DIR = os.path.join(SPAM_PATH, "spam")
ham_filenames = [name for name in sorted(os.listdir(HAM_DIR)) if len(name) > 20]
spam_filenames = [name for name in sorted(os.listdir(SPAM_DIR)) if len(name) > 20]
        

import email
import email.policy

def load_email(is_spam, filename, spam_path=SPAM_PATH):
    directory = "spam" if is_spam else "easy_ham"
    with open(os.path.join(spam_path, directory, filename), "rb") as f:
        return email.parser.BytesParser(policy=email.policy.default).parse(f)
    

ham_emails = [load_email(is_spam=False, filename=name) for name in ham_filenames]
spam_emails = [load_email(is_spam=True, filename=name) for name in spam_filenames]


print(ham_emails[1].get_content().strip())