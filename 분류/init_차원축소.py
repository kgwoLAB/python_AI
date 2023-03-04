from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
mnist.keys()

X, y = mnist["data"], mnist["target"]

X.shape

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

# 매니폴드 학습
# 스위스롤 처럼 데이터의 공간이 뒤틀려 있을때 그대로 평면에 퉁영시키면 서로 뭉개진다. 
# 그러기에 스위스롤을 펼치고자 합니다. 

# d차원 초평면으로 보일 수 있는 n 차원 공간의 일부
# 차원을 감소시키면 훈련 속도가 빨라지지만 항상 더 낫거나 간단한 솔루션이 되는건 데이터 셋 맘이다.


# PCA 주성분 분석
# 분산 보존
# 저 차원의 초평면에 훈련 세트를 투영하기 전 먼저 올바른 초평면을 선택해야 합니다. 
# PCA는 분산이 최대인 축을 찾습니다. 또한 첫 번쨰 축에 직교하고 남은 분산을 최대한 보존하는 두 번째 축을 찾습니다. 
# 특잇값 분해 SVD라는 표준 행렬 분해 기술로 해결함.

# 해당 코드는 넘파이 svd() 함수를 사용해 훈련 세트의 모든 주성분을 구한 후 처음 두 개의 PC를 정의하는 두 개의 단위 벡터를 추출합니다. 
X_centered = X - X.mean(axis=0)
U, s, Vt = np.linalg.svd(X_centered)
c1 = Vt.T[:,0]
c2 = Vt.T[:,1]

W2 = Vt.T[:, :2]
X2D = X_centered.dot(W2)



# 사이킷런으로 구현하기
from sklearn.decomposition import PCA
import numpy as np
pca = PCA(n_components=2) # 2차원으로 축소하겠다는 의미
X2D = pca.fit_transform(X)
pca.explained_variance_ratio_# 설명된 분산의 비율

# 적절한 차원 수 선택하기
pca = PCA()
pca.fit(X_train)
cumsum = np.cumsum(pca.explained_variance_ratio_)
d = np.argmax(cumsum >= 0.95) +1

plt.figure(figsize=(6,4))
plt.plot(cumsum, linewidth=3)
plt.axis([0, 400, 0, 1])
plt.xlabel("Dimensions")
plt.ylabel("Explained Variance")
plt.plot([d, d], [0, 0.95], "k:")
plt.plot([0, d], [0.95, 0.95], "k:")
plt.plot(d, 0.95, "ko")
plt.annotate("Elbow", xy=(65, 0.85), xytext=(70, 0.7),
             arrowprops=dict(arrowstyle="->"), fontsize=16)
plt.grid(True)
#save_fig("explained_variance_plot")
plt.show()


pca = PCA(n_components=d)
X_reduced = pca.fit_transform(X_train)
cumsum = np.cumsum(X_reduced)
