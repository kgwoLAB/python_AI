# 데이터는 이미 설치되있으니까 걍 쓰자

import tensorflow as tf
from tensorflow import keras
import pandas as pd

housing = pd.read_csv("E:\Git\project-python\분류\datasets\housing\housing.csv")

housing.info()

import matplotlib.pyplot as plt
housing.hist(bins=50, figsize=(20,15))


# 데이터 정제부터하자

from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.impute import SimpleImputer


from sklearn.model_selection import StratifiedShuffleSplit

housing["income_cat"] = pd.cut(housing["median_income"], bins=[0,1.5,3.0,4.5,6, np.inf],labels=[1,2,3,4,5])
housing["income_cat"].hist()

from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)





# 1.3.3 Stratified ShuffleSplit
from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]


# 소득 카테고리의 비율 측정
strat_test_set["income_cat"].value_counts() / len(strat_test_set)

# 불필요한 데이터 삭제
for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)


housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()
housing.info()
housing_num = housing.drop("ocean_proximity", axis=1)
housing_string = housing["ocean_proximity"].copy()

from sklearn.base import BaseEstimator, TransformerMixin

rooms_ix, bedrooms_ix, population_ix, households_ix = 3,4,5,6
class split_IT(BaseEstimator, TransformerMixin):
    def __init__(self,n_splits=1,test_size=0.2,random_state=42):
        self.n_splits = n_splits
        self.test_size = test_size
        self.random_state = random_state
        
    def fit(self, X,y=None):
        return self
    
    def transform(self, X):
        from sklearn.model_selection import StratifiedShuffleSplit
        train, test = StratifiedShuffleSplit(n_splits=self.n_splits, test_size=self.test_size, random_state=self.random_state)

        train_x = train.drop("median_house_value", axis=1)
        train_y = train["median_house_value"].copy()
        
        return train_x, train_y;
    
    
    
class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):
        self.add_bedrooms_per_room = add_bedrooms_per_room
    
    def fit(self,X,y=None):
        return self
    def transform(self,X):
        rooms_per_household =X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        
        else:
            return np.c_[X, rooms_per_household, population_per_household]


from sklearn.preprocessing import StandardScaler

first_PL = Pipeline([
  ("imputer",SimpleImputer(strategy="median")),
  ("AddAtrributer",CombinedAttributesAdder(add_bedrooms_per_room=True)),
  ("scaler",StandardScaler()),
])

import numpy as np

first_PL_test = pd.DataFrame(first_PL.fit_transform(housing_num), columns=list(housing_num.columns)+["rooms_per_household","population_per_household","bedrooms_per_room"], index=housing_num.index)

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# 데이터가 많은 것끼리 데이터 정제

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

full_PL = ColumnTransformer([
    ("num", first_PL, list(housing_num)),
    ("cat", OneHotEncoder(), ["ocean_proximity"]),
])

full_PL_tr= full_PL.fit_transform(housing)
housing_num.info()
pd.DataFrame(full_PL_tr).info()
housing_labels.info()

housing.info()
A = housing.corr()
A["median_house_value"].sort_values(ascending=False)
housing["ocean_proximity"].value_counts()

# 데이터 가공 완뇨

#학습시작
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(full_PL_tr, housing_labels)

# 테스트 5개 추출
some_data = housing_num.iloc[:5]
some_label = housing_labels.iloc[:5]

some_data.info()
some_data_prepared = full_PL.fit_transform(some_data)
lin_reg.predict(some_data_prepared)