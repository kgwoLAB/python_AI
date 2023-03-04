"""_summary_
군집 
비슷한 샘플을 클러스트로 모읍니다. 
군집은 데이터 분석, 고객 분류, 추천 시스템, 검색 엔진, 이미지 분할, 준지도 학습, 차원 축소 등에 사용할 수 있는 훌륭한 도구입니다.

이상치 탐지
정상 데이터가 어떻게 보이는지 학습합니다. 그 다음 비정상 샘플을 감지하는 데 사용합니다. 예를 들면
제조 라인에서 결함 제품을 감지하거나 시계열 데이터에서 새로운 트랜드를 찾습니다.

밀도 추정
데이터셋 생성 확률 과정의 확률 밀도 함수를 추정합니다.
밀도 추정은 이상치탐지에서 널리 사용됩니다. 밀도가 매우 낮은 영역에 놓인 샘플이 이상치일 가능성이 높습니다.
또한 데이터 분석과 시각화에 유용합니다.


#군집 Cluster

다양한 애플리케이션에서 사용됩니다.
고객분류 - 추천 시스템
데이터 분석 - 새로운 데이터셋을 분석할때 군집 알고리즘을 실행하고 각 클러스터를 따로 분석하면 도움이 됩니다.
차원 축소 기법 - 각 클러스터에 대한 샘플 친화성을 측정할 수 있습니다.  k개의 클러스터가있다면 k개의 차원이라함
이상치 탐지 - 부정 거래 감지 
준지도 학습 = 동일한 클러스터에 있는 모든 샘플에 레이블을 전파할수 있음
검색 엔진 - 
이미지 분할 - 물체의 윤곽을 감지하기 쉬워져 물체 탐지 및 추적 시스템에서 많이 사용됨.





k-평균 - 로이드 포지 알고리즘
몇 번의 반복을 통해 데이터셋을 빠르고 효율적으로 클러스터로 묶습니다.
각 클러스터의 중심을 찾고 가장 가까운 클러스터에 샘플을 할당합니다.
"""
import matplotlib as mpl
from sklearn.datasets import load_iris
data = load_iris()
X = data.data
y = data.target
data.target_names


import matplotlib.pyplot as plt
plt.figure(figsize=(9, 3.5))

plt.subplot(121)
plt.plot(X[y==0, 2], X[y==0, 3], "yo", label="Iris setosa")
plt.plot(X[y==1, 2], X[y==1, 3], "bs", label="Iris versicolor")
plt.plot(X[y==2, 2], X[y==2, 3], "g^", label="Iris virginica")
plt.xlabel("Petal length", fontsize=14)
plt.ylabel("Petal width", fontsize=14)
plt.legend(fontsize=12)
plt.subplot(122)
plt.scatter(X[:, 2], X[:, 3], c="k", marker=".")
plt.xlabel("Petal length", fontsize=14)
plt.tick_params(labelleft=False)
plt.show()




# 버블로 예제를 만들기
import numpy as np
from sklearn.datasets import make_blobs
blob_centers = np.array(
    [[ 0.2,  2.3],
     [-1.5 ,  2.3],
     [-2.8,  1.8],
     [-2.8,  2.8],
     [-2.8,  1.3]])
blob_std = np.array([0.4, 0.3, 0.1, 0.1, 0.1])

X, y = make_blobs(n_samples=2000, centers=blob_centers,
                  cluster_std=blob_std, random_state=7)


def plot_clusters(X, y=None):
    plt.scatter(X[:, 0], X[:, 1], c=y, s=1)
    plt.xlabel("$x_1$", fontsize=14)
    plt.ylabel("$x_2$", fontsize=14, rotation=0)

plt.figure(figsize=(8, 4))
plot_clusters(X)
plt.show()



from sklearn.cluster import KMeans
k = 5
kmeans = KMeans(n_clusters=k)
y_pred = kmeans.fit_predict(X)

# K평균이 분배해준 라벨을 확인할 수 있음.
y_pred
kmeans.labels_

# 샌트로이드, 특정 중심을 의미함.
kmeans.cluster_centers_


# 그냥 혼자 그려봄
def plot_data(X):
    plt.plot(X[:,0], X[:,1], 'k.', markersize=2)

def plot_centroids(centroids, weights=None, circle_color='w', cross_color='k'):
    if weights is not None:
        centroids = centroids[weights > weight.max() / 10]
        
    plt.scatter(centroids[:,0],centroids[:,1], marker='o', s=35, linewidths=8,color=circle_color, zorder=10, alpha=0.9)
    plt.scatter(centroids[:,0],centroids[:,1], marker='x', s=2, linewidths=12,color=circle_color, zorder=11, alpha=1)
    
def plot_decision_boundaries(cluseterer, X, resolution=1000, show_centroids=True, show_xlabels=True, show_ylabels=True):
    mins = X.min(axis=0) - 0.1
    maxs = X.max(axis=0) + 0.1
    
    xx, yy = np.meshgrid(np.linspace(mins[0], maxs[0], resolution),np.linspace(mins[1], maxs[1], resolution))
    
    Z = cluseterer.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.contourf(Z, extent=(mins[0], maxs[0], mins[1],maxs[1]),cmap="Pastel2")
    plt.contour(Z, extent=(mins[0], maxs[0], mins[1],maxs[1]),linewidths=1, colors='k')
    
    
    plot_data(X)
    if show_centroids:
        plot_centroids(cluseterer.cluster_centers_)
    if show_xlabels:
        plt.xlabel("$x_1$", fontsize=14)
    else:
        plt.tick_params(labelbottom=False)
    if show_ylabels:
        plt.ylabel("$x_2$", fontsize=14, rotation=0)
    else:
        plt.tick_params(labelleft=False)
        
        
plt.figure(figsize=(8, 4))
plot_decision_boundaries(kmeans, X)
plt.show()


# 새로운 샘플에 가장가까운 센트로이드의 클러스터를 할당할 수 있음
X_new = np.array([[0, 2], [3, 2], [-3, 3], [-3, 2.5]])
kmeans.predict(X_new)


kmeans.transform(X_new)
"""_summary_
하드 군집이라는 샘플을 하나의 클러스터에 할당하는 것보다 클러스터마다 샘플에 점수를 부여하는 것이 유용할 수 있습니다.
이를 소프트 군집이라고 합니다.

해당점수는 샘플과 센트로이드 사이의 거리가 될수 있습니다.
반대로 가우시안 방사기저 함수와같은 유사도 점수가 될수도있습니다.

"""





# k 평균 알고리즘을 1,2,3회 반복한것을 확인해보기
kmeans_iter1 = KMeans(n_clusters=5, init="random", n_init=1,
                      algorithm="full", max_iter=1, random_state=0)
kmeans_iter2 = KMeans(n_clusters=5, init="random", n_init=1,
                      algorithm="full", max_iter=2, random_state=0)
kmeans_iter3 = KMeans(n_clusters=5, init="random", n_init=1,
                      algorithm="full", max_iter=3, random_state=0)
kmeans_iter1.fit(X)
kmeans_iter2.fit(X)
kmeans_iter3.fit(X)

plt.figure(figsize=(10, 8))

plt.subplot(321)
plot_data(X)
plot_centroids(kmeans_iter1.cluster_centers_, circle_color='r', cross_color='w')
plt.ylabel("$x_2$", fontsize=14, rotation=0)
plt.tick_params(labelbottom=False)
plt.title("Update the centroids (initially randomly)", fontsize=14)

plt.subplot(322)
plot_decision_boundaries(kmeans_iter1, X, show_xlabels=False, show_ylabels=False)
plt.title("Label the instances", fontsize=14)

plt.subplot(323)
plot_decision_boundaries(kmeans_iter1, X, show_centroids=False, show_xlabels=False)
plot_centroids(kmeans_iter2.cluster_centers_)

plt.subplot(324)
plot_decision_boundaries(kmeans_iter2, X, show_xlabels=False, show_ylabels=False)

plt.subplot(325)
plot_decision_boundaries(kmeans_iter2, X, show_centroids=False)
plot_centroids(kmeans_iter3.cluster_centers_)

plt.subplot(326)
plot_decision_boundaries(kmeans_iter3, X, show_ylabels=False)

plt.show()

# 위코드를 살펴보면 센트로이드의 초기화에 따라 군집이 나눠지는게 보인다.
# 그러기에 센트로이드는 잘 초기화 해서 진행해야한다.


# 센트로이드 초기화 방법
# 센트로이드 위치를 근사하게 알수 있다면 init 매개변수에 넘파이 배열을 지정하고 n_init=1로 설정할수 있습니다.
good_init = np.array([[-3,3],[-3,2],[-3,1],[-1,2],[0,2]])
kmeans = KMeans(n_clusters=5, init=good_init, n_init=1)

# 또 다른 방법은 랜덤 초기화를 다르게 하여 여러 번 알고리즘을 실행하고 가장 좋은 솔루션을 선택하는 것입니다.
# 랜덤 초기화 횟수는 n_init 매개변수로 조절하며 기본값은 10입니다. 
# 최선의 솔루션은 샘플과 가장 가까운 센트로이드 사이의 제곱 거리 합인 이너셔라는 성능 지표를 사용합니다.
kmeans.inertia_
kmeans.score(X)




kmeans_per_k = [KMeans(n_clusters=k, random_state=42).fit(X)
                for k in range(1, 10)]
inertias = [model.inertia_ for model in kmeans_per_k]
plt.figure(figsize=(8, 3.5))
plt.plot(range(1, 10), inertias, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Inertia", fontsize=14)
plt.annotate('Elbow',
             xy=(4, inertias[3]),
             xytext=(0.55, 0.55),
             textcoords='figure fraction',
             fontsize=16,
             arrowprops=dict(facecolor='black', shrink=0.1)
            )
plt.axis([1, 8.5, 0, 1300])
plt.show()

"""
score가 음수값을 반환하는 이유는 큰 값이 좋은 것이다 라는 규칙을 따라야 하는 것입니다.     


k-평균 속도 개선과 미니배치 k-평균
불필요한 거리 계싼을 많이 피함으로써  알고리즘 속도를 상당히 높일 수 있음 - 삼각 부등식(엘칸)
해당 알고리즘은 KMeans 클래스에서 기본으로 사용하며 algorithm =full로 원래 지정했어야 했음

# 2010년 데이비드 스컬리는 전체 데이터셋을 사용해 반복하지 않고 이 알고리즘은 각 반복마다 미니배치를 사용해 센트로이드를 조금씩 이동합니다.
이는 일반적으로 3배~4배 정도 속도를 올립니다.
MiniBatchKMeans





# 최적의 클러스터 개수 찾기
가장작은 이너셔를 가진 모델을 선택하는건 
이너셔는 k가 증가함에 따라 점점 작이지므로 k를 선택할 때 좋은 성능 지표는 아닙니다. 
답을 모를떄는 4가 좋은 선택이 됩니다. 

더 정확한 방법은 실루엣 점수입니다.
모든 샘플에 대한 실루엣 계수의 평균입니다. 계산 방법은 (b-a) / max(a,b) 
a는 동일한 클러스터에 있는 다른 샘플까지 평균거리, b는 가장 가따운 클러스터 까지 평균 거리 
    """


from sklearn.metrics import silhouette_score
silhouette_score(X, kmeans.labels_)

silhouette_scores = [silhouette_score(X, model.labels_)
                     for model in kmeans_per_k[1:]]

plt.figure(figsize=(8, 3))
plt.plot(range(2, 10), silhouette_scores, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Silhouette score", fontsize=14)
plt.axis([1.8, 8.5, 0.55, 0.7])
plt.show()


# 실루엣 다이어그램으로도 확인할수 있습니다.
# 높이는 클러스터가 포함하고 있는 샘플의 개수를 의미하며 너비는 클러스터에 포함된 샘플의 정렬된 실루옛계수
# 너비가 넓을수록 좋음?

from sklearn.metrics import silhouette_samples
from matplotlib.ticker import FixedLocator, FixedFormatter


plt.figure(figsize=(11, 9))

for k in (3, 4, 5, 6):
    plt.subplot(2, 2, k - 2)
    
    y_pred = kmeans_per_k[k - 1].labels_
    silhouette_coefficients = silhouette_samples(X, y_pred)

    padding = len(X) // 30
    pos = padding
    ticks = []
    for i in range(k):
        coeffs = silhouette_coefficients[y_pred == i]
        coeffs.sort()

        color = mpl.cm.Spectral(i / k)
        plt.fill_betweenx(np.arange(pos, pos + len(coeffs)), 0, coeffs,
                          facecolor=color, edgecolor=color, alpha=0.7)
        ticks.append(pos + len(coeffs) // 2)
        pos += len(coeffs) + padding

    plt.gca().yaxis.set_major_locator(FixedLocator(ticks))
    plt.gca().yaxis.set_major_formatter(FixedFormatter(range(k)))
    if k in (3, 5):
        plt.ylabel("Cluster")
    
    if k in (5, 6):
        plt.gca().set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
        plt.xlabel("Silhouette Coefficient")
    else:
        plt.tick_params(labelbottom=False)

    plt.axvline(x=silhouette_scores[k - 2], color="red", linestyle="--")
    plt.title("$k={}$".format(k), fontsize=16)

plt.show()




"""_summary_
k - 평균의 한계
장점 : 속도가 빠르며 확장이 용이합니다.
단점? : 최적이 아닌 솔루션을 피하려면 알고리즘을 여러 번 실행해야하며 클러스터 개수를 지정해 줘야합니다.
클러스터의 크기나 밀집도가 서로 다르거나 원형이 아닌 경우 잘 작동하지 않습니다.

스케일링을 해줘야한다.

따라서 데이터에 따라 잘 수행할수 있는 군집 알고리즘이 다릅니다. 




# 군집을 사용한 이미지 분할
이미지를 세그먼트 여러 개로 분할하는 작업입니다.
시맨틱 분할에서는 동일한 종류의 물체에 속한 모든 픽셀은 같은 세그먼트에 할당됩니다. 

만약, 자율 주행 자동차의 비전 시스템에서 보행자 이미지를 구성하는 모든 픽셀은 보행자 세그먼트에 할당될것입니다. 
합성곱 신경망?
색상 분할만 수행할거임
    """
    
    

# 무당벌레 이미지를 다운로드합니다
import os, urllib


PROJECT_ROOT_DIR = os.path.join("datasets")

images_path = os.path.join(PROJECT_ROOT_DIR, "images", "unsupervised_learning")
os.makedirs(images_path, exist_ok=True)
DOWNLOAD_ROOT = "https://raw.githubusercontent.com/rickiepark/handson-ml2/master/"
filename = "ladybug.png"
print("Downloading", filename)
url = DOWNLOAD_ROOT + "images/unsupervised_learning/" + filename
urllib.request.urlretrieve(url, os.path.join(images_path, filename))

from matplotlib.image import imread
image = imread(os.path.join(images_path, filename))
image.shape

# 해당 이미지는 3d 배열로 표현됩니다.
# 첫 차원은 높이 둘은 너비 셋은 커널 채널 개수 입니다. rgb임 0~1

X = image.reshape(-1,3)
kmeans = KMeans(n_clusters=8).fit(X)
segmented_img = kmeans.cluster_centers_[kmeans.labels_]
segmented_img = segmented_img.reshape(image.shape)

segmented_imgs = []
n_colors = (10, 8, 6, 4, 2)
for n_clusters in n_colors:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(X)
    segmented_img = kmeans.cluster_centers_[kmeans.labels_]
    segmented_imgs.append(segmented_img.reshape(image.shape))
    
plt.figure(figsize=(10,5))
plt.subplots_adjust(wspace=0.05, hspace=0.1)

plt.subplot(231)
plt.imshow(image)
plt.title("Original image")
plt.axis('off')

for idx, n_clusters in enumerate(n_colors):
    plt.subplot(232 + idx)
    plt.imshow(segmented_imgs[idx])
    plt.title("{} colors".format(n_clusters))
    plt.axis('off')



# 군집을 사용한 전처리
# 군집은 지도 학습 알고리즘을 적용하기 전에 전처리 단계로 사용할 수 있습니다.
# 차원 축소에 군집을 사용하는 예를 위해 숫자 데이터 셋을 다루어 보겠습니다.

from sklearn.datasets import load_digits
X_digits, y_digits = load_digits(return_X_y=True)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_digits, y_digits, test_size=0.2, random_state=42)

from sklearn.linear_model import LogisticRegression

log_reg =LogisticRegression()
log_reg.fit(X_train,y_train)

log_reg.score(X_test, y_test)



from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ("kmeans", KMeans(n_clusters=50, random_state=42)),
    ("log_reg", LogisticRegression(multi_class="ovr", solver="lbfgs", max_iter=5000, random_state=42)),
])
pipeline.fit(X_train, y_train)
pipeline.fit(X_train, y_train)

pipeline.score(X_test,y_test)




# 교차 검증을 통해 최적의 클러스터를 찾아볼수도있다.
from sklearn.model_selection import GridSearchCV

param_grid = dict(kmeans__n_clusters=range(2, 100))
grid_clf = GridSearchCV(pipeline, param_grid, cv=3, verbose=2)
grid_clf.fit(X_train, y_train)

grid_clf.best_params_
grid_clf.score(X_test, y_test)





# 군집을 사용한 준지도 학습
# 레이블이 없는 데이터가 많고 레이블이 있는 데이터가 적을 때 사용합니다. 
n_labeled =50
log_reg =LogisticRegression()
log_reg.fit(X_train[:n_labeled], y_train[:n_labled])
log_reg.score(X_test, y_test)

# 개선하는 방법?  50개의 클러스터로 모읍니다.
# 그다음 각 클러스터에서 센트로이드에 가장가까운이미지를 찾습니다. 이를 대표 이미지라고 합니다.
k =50
kmeans =KMeans(n_clusters=50)
X_digits_dist = kmeans.fit_transform(X_train)
representative_digit_idx = np.argmin(X_digits_dist, axis=0)
X_representative_digits = X_train[representative_digit_idx]


plt.figure(figsize=(8, 2))
for index, X_representative_digit in enumerate(X_representative_digits):
    plt.subplot(k // 10, 10, index + 1)
    plt.imshow(X_representative_digit.reshape(8, 8), cmap="binary", interpolation="bilinear")
    plt.axis('off')

plt.show()



# 이미지를 보고 수동으로 레이블 할당해보기

y_train[representative_digit_idx]
y_representative_digits = np.array([
    0, 1, 3, 2, 7, 6, 4, 6, 9, 5,
    1, 2, 9, 5, 2, 7, 8, 1, 8, 6,
    3, 1, 5, 4, 5, 4, 0, 3, 2, 6,
    1, 7, 7, 9, 1, 8, 6, 5, 4, 8,
    5, 3, 3, 6, 7, 9, 7, 8, 4, 9])



# 레이블 전파 
y_train_propagated = np.empty(len(X_train), dtype=np.int32)
for i in range(k):
    y_train_propagated[kmeans.labels_==i] = y_representative_digits[i]
    
    
    
percentile_closest = 75

X_cluster_dist = X_digits_dist[np.arange(len(X_train)), kmeans.labels_]
for i in range(k):
    in_cluster = (kmeans.labels_ == i)
    cluster_dist = X_cluster_dist[in_cluster]
    cutoff_distance = np.percentile(cluster_dist, percentile_closest)
    above_cutoff = (X_cluster_dist > cutoff_distance)
    X_cluster_dist[in_cluster & above_cutoff] = -1
    
partially_propagated = (X_cluster_dist != -1)
X_train_partially_propagated = X_train[partially_propagated]
y_train_partially_propagated = y_train_propagated[partially_propagated]


log_reg.fit(X_train_partially_propagated, y_train_partially_propagated)
log_reg.score(X_test, y_test)






# DBSCAN 밀집된 연속적 지역을 클러스터로 정의합니다. 