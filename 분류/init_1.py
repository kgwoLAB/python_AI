# 1. 데이터
import os
import tarfile
import urllib
import pandas as pd
import numpy as np

# https://wikidocs.net/92112
# 1.1 데이터 가져오기
DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()
    
def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)

fetch_housing_data()
housing = load_housing_data()


# 1.2 데이터 확인하기

housing.head() # 앞부분
housing.info() # 간단한 정보
housing["ocean_proximity"].value_counts() # 카테고리 확인
housing.describe() # 요약정보


%matplotlib inline # 주피터 노트북의 매직 명령어
import matplotlib.pyplot as plt
housing.hist(bins=50, figsize=(20,15))
plt.show()



# 데이터가 많은 것끼리 데이터 정제
housing["income_cat"] = pd.cut(housing["median_income"], bins=[0,1.5,3.0,4.5,6, np.inf],labels=[1,2,3,4,5])
housing["income_cat"].hist()





housing.shape

# 1.3 테스트 세트 나누기

# 1.3.1 - 1.3.2 : 순수한 무작위 샘플링 방식
# 1.3.3 : 특별한 무작위 샘플링 방식

# 1.3.1 numpy 를 사용해서 나누기
import numpy as np

def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data)*test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

train_set, test_set = split_train_test(housing, 0.2)

# p86을 보면 해시값으로 최대값의 20-%작거나 같은 샘플만 테스트 세트로 보내는 방식이있다.
#이는 데이터세이 갱신되더라도 테스트 세트가 동일?하게 유지됩니다.

# 1.3.2 sklearn train_test_split 사용하기
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
    set_.drop("income", axis=1, inplace=True)



# 그래프로 그려보기
import matplotlib.pyplot as plt

housing = strat_train_set.copy()
housing.plot(kind="scatter", x="longitude", y="latitude")
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)

housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
             s=housing["population"]/100, label="population", figsize=(10,7),
             c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
             sharex=False)



# 빨간색은 높은 가격, 파란색은 낮은가격, 큰원은 인구가 밀집된 구역 

# 2.4.2 상관관계 조사
# 모든 특성 간의 표준 상관계수를 corr() 메서드를 이용해 쉽게 계산가능

corr_matrix = housing.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)


from pandas.plotting import scatter_matrix
attributes = ["median_house_value", "median_income", "total_rooms","housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12,8))



# 2.4.3 특성 조합 실험
housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"]=housing["population"]/housing["households"]

corr_matrix = housing.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)





# 2.5 머신러닝 알고리즘을 위한 데이터 준비
# 기존데이터 복구
housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()


# 2.5.1 데이터 정제
housing.info()

housing.dropna(subset=["total_bedrooms"])
housing.drop("total_bedrooms", axis=1)
median = housing["total_bedrooms"].median()
housing["total_bedrooms"].fillna(median,inplace=True)




# SimleImputer는 누락된 값을 손쉽게 다루어 줍니다.
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="median")
housing_num = housing.drop("ocean_proximity", axis=1)
imputer.fit(housing_num)
housing_num.median()
housing_num.head()

X = imputer.transform(housing_num)
housing_tr = pd.DataFrame(X, columns=housing_num.columns, index=housing_num.index)
housing_tr


"""_summary_
사이킷런의 철학?

일관성
 - 추정기 estimator
 - 변환기 transformer   transform(), fit_transform()
 - 예측기 predictor     predict(), score()
 - 검사기능 -
 - 클래스 남용 방지 - 넘파이 배열이나 사이파이 희소 행렬로 표현
 - 조합성 - 기존의 구성요소를 최대한 사용 Pipline
 - 합리적 기본값 
"""

# 2.5.2 텍스트와 범주형 특성 다루기
housing_cat = housing[["ocean_proximity"]]
housing_cat.head(10)

# 숫자로 변환하기
from sklearn.preprocessing import OrdinalEncoder
ordinal_encoder = OrdinalEncoder()
housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)
housing_cat_encoded[:10]
ordinal_encoder.categories_

# 원핫 인코딩 

from sklearn.preprocessing import OneHotEncoder
cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
housing_cat_1hot

housing_cat_1hot.toarray()



# 2.5.3 나만의 변환기
from sklearn.base import BaseEstimator, TransformerMixin

rooms_ix, bedrooms_ix, population_ix, households_ix = 3,4,5,6

class CombinedAtrributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):
        self.add_bedrooms_per_room = add_bedrooms_per_room
    
    def fit(self, X, y=None):
        return self
    def transform(self,X):
        rooms_per_household =X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        
        else:
            return np.c_[X, rooms_per_household, population_per_household]
        

attr_adder = CombinedAtrributesAdder(add_bedrooms_per_room=True)
housing_extra_attribs = attr_adder.transform(housing.values)

# 2.5.4 특성 스케일
# min-max 스케일링과 표준화가 널리 사용됩니다.
# 0 ~ 1 범위에 들도록 정규화를 한다.
# 표준화는 평균을 뺀 후 표준편차로 나누어 분포의 분산이 1이 되도록 합니다. 

from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
ss.fit(housing_num)
housing_ss = pd.DataFrame(ss.transform(housing_num),columns=housing_num.columns, index=housing_num.index)
housing_ss.hist()


# 2.5.5 변환 파이프라인
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('attribs_adder', CombinedAtrributesAdder()),
    ('std_scaler', StandardScaler()),
])
# 데이터를 빈곳을 채우고 만들고 스케일을 처리했으면 남은건?
housing_num_tr = num_pipeline.fit_transform(housing_num)


from sklearn.compose import ColumnTransformer

num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

full_pipline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", OneHotEncoder(), cat_attribs),
])


housing.columns
housing_num.columns
test_num_tr = pd.DataFrame(full_pipline.fit_transform(housing))
test_num_tr.head()










# 2.6 모델 선택과 훈련
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(test_num_tr, housing_labels)


# 테스트 5개 추출
some_data = housing.iloc[:5]
some_label = housing_labels.iloc[:5]

# 예측하기
some_data_prepared = full_pipline.transform(some_data)
lin_reg.predict(some_data_prepared)
some_label

# mean squere error 함수를 사용해 전체 훈련 세트에 대한 RMSE 오차 뭐시기를 측정해보기
from sklearn.metrics import mean_squared_error
housing_predictions = lin_reg.predict(test_num_tr)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
lin_rmse # 오차가 너무크네





# 다른걸로 훈련해보기
from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor()
tree_reg.fit(test_num_tr, housing_labels)
housing_predictions = tree_reg.predict(test_num_tr)
tree_mse = mean_squared_error(housing_labels,housing_predictions)
tree_rmse = np.sqrt(tree_mse)
tree_rmse # 오차가 전혀없네 ..






# 2.6.2 교차 검증을 사용한 평가
# 결정 트리 모델을 평가하는 방법
# 사이킷런의 k겹 교차 검증. 폴드라고 불리는 10개의 서브셋 무작위분할, 매번 다른 폴드 선택해 평가에 사용됨 
from sklearn.model_selection import cross_val_score
scores = cross_val_score(tree_reg, test_num_tr, housing_labels)
tree_rmse_scores = np.sqrt(-scores)

scores = cross_val_score(lin_reg, test_num_tr, housing_labels, scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-scores)
display_scores(tree_rmse_scores)


# 결과 살펴보기
def display_scores(scores):
    print("점수", scores)
    print("평균:",scores.mean())
    print("표준편차",scores.std())

display_scores(scores)







# 랜덤 포레스트
from sklearn.ensemble import RandomForestRegressor
# 앙상블 학습은 무작위로 특성을 선택해 결정 트리로 만들고 그 예측영균내는 방식, 여러 다른 모델을 모아 하나로만든다?
forest_Reg = RandomForestRegressor()
forest_Reg.fit(test_num_tr,housing_labels)

forest_scores = cross_val_score(forest_Reg, test_num_tr, housing_labels, scoring="neg_mean_squared_error", cv=10)
forest_rmse_scores = np.sqrt(-forest_scores)
display_scores(forest_rmse_scores)



# 모델을 쉽게 저장및 불러오기
# pickle도 좋은방법임
import joblib
joblib.dump(my_model, "my_model.pkl")
my_model_loaded = joblib.load("my_model.pk1")





# 2.7 모델 세부 튜닝
# 2.7.1 그리드 탐색
# 가장 단순함. GridSearchCV

from sklearn.model_selection import GridSearchCV

param_grid = [
    {'n_estimators':[3,10,30], 'max_features':[2,4,6,8]},
    {'bootstrap':[False], 'n_estimators':[3,10], 'max_features':[2,3,4]},
]

forest_reg = RandomForestRegressor()

# cv는 몇차 검증을 할건인지 정하는거임
grid_search = GridSearchCV(forest_reg, param_grid, cv=5, scoring='neg_mean_squared_error', return_train_score=True)
grid_search.fit(test_num_tr, housing_labels)

grid_search.best_params_
# 하이퍼피라미터 값을 찾을떄는 연속된 10의 거듭제곱 수로 시도는게 좋음
grid_search.best_estimator_
# 최적의 추정기에 직접 접근 가능
# 평가 점수도 확인 가능


cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"],cvres["params"]):
    print(np.sqrt(-mean_score), params)
    
    # 특성을 다루거나 특성 선택 등을 처리에 사용
    
    

# 2.7.2 랜덤 탐색
# RandomizedSearchCV
from sklearn.model_selection import RandomizedSearchCV
param = {
    'n_estimators':range(1,50),
    'max_features':range(1,20),
}
Random_search = RandomizedSearchCV(forest_reg,param_distributions=param, cv=5, scoring='neg_mean_squared_error', return_train_score=True, n_jobs=-1)
Random_search.fit(test_num_tr, housing_labels)
Random_search.best_params_

"""
> cv_results_ : 파라미터 조합별 결과 조회
> best_params_ : 가장 좋은 성능을 낸 parameter 조합 조회
> best_estimator_ : 가장 좋은 성능을 낸 모델 반환
"""

# 앙상블방법 / 최상의 모델을 연결해보는 것


# 2.7.4 최상의 모델과 오차 분석
# 각 특성의 상대적인 중요도를 파악
feature_importances = grid_search.best_estimator_.feature_importancescat_
feature_importances
extra_attribs = ["rooms_per_hhold", "pop_per_hhold", "bedrooms_per_room"]
cat_encoder = full_pipline.named_transformers_["cat"]
cat_one_hot_attribs = list(cat_encoder.categories_[0])
attributes = num_attribs + extra_attribs + cat_one_hot_attribs
sorted(zip(feature_importances, attributes), reverse=True)
# 추가 특성을 포함하거나 불필요한 특성제거, 이상치 제외



# 2.7.5 테스트 세트로 시스템 평가하기
final_model = grid_search.best_estimator_
final_model = Random_search.best_estimator_

X_test = strat_test_set.drop("median_house_value",axis=1)
y_test = strat_test_set["median_house_value"].copy()

X_test_prepared =full_pipline.transform(X_test)

final_predictions = final_model.predict(X_test_prepared)

final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
final_rmse


# 2.8 론칭 모니터링, 시스템 유지보수

"""_summary_
모델을 상용 환경에 배포하는 방법중 한가지는
전체 전처리 파이프라인과 예측 파이프라인이 포함된 사이킷런 모델을 저장함
그다음 훈련된 모델을 상용 환경에 로드하고 predict 메서드를 호출해 예측을 만들기


웹 어플리케이션 REST API를 통해 질의할수 있는 전용 웹서비스로 모델을 감쌀수있다.

이렇게 한다면 주 애플리케이션을 건들지 않고 모델을 새버전으로 업그레이드 하기 쉽습니다.


# 구글 클라우드 AI 플랫폼과 같은 클라우드에 배포하는 방식도있습니다.
이를 사용해 모델을 저장하고 GCS 구글 클라우드 스토리지에 업로드합니다.

로드 밸런싱과 자동 확장을 처리하는 간단한 웹서비스 JSON~

시스템의 실시간 성능 체크, 성능이 떨어졌을경우 알람을 통지받을수 있어야 함.


데이터가 변하게되면 낙후하게 됩니다.
이를 해결하기 위해서는
모델업데이트를 자동화 해야한다.

정기적으로 새로운 데이터를 수집하고 레이블을 답니다.
모델을 훈련하고 하이퍼파라미터를 자동으로 세부 튜닝하는 스크립트를 작성합니다.
업데이트된 테스트 세트에서 새로운 모델과 이전 모델을 평가하는 스크립트를 하나더 작성합니다.
성능이 감소하지 않으면 새로운 모델을 제품에 배포합니다.


모델의 입력 데이터 품질을 평가해야함. 나쁜 품질의 신호로 성능 감소가 발생할 수있음.

백업도 필수


"""



# test
from sklearn.svm import SVR

clf = SVR(C=10, kernel='linear', epsilon=0.2)
clf.fit(test_num_tr,housing_labels)
clf.predict()
clf_scores = cross_val_score(clf, test_num_tr, housing_labels, scoring="neg_mean_squared_error", cv=10)
clf_rmse_scores = np.sqrt(-clf_scores)
display_scores(clf_rmse_scores)

