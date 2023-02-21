import numpy as np

# 행령 / 배열 처리 및 연산
# 난수생성

# 1. 배열 생성
###############
# 일반 배열 생성
###############
a = [[1,2,3],[4,5,6]] # 리스트에서 행렬생성
b = np.array(a) 
print(b)

print(b.ndim) # 배열의 열수(차원)

print(b.shape) # 배열의 차원

print(b[0,0]) # 배열의 원소 접근


#################
# 특수한 배열 생성
#################
print(np.arange(10)) # 1씩 증가하는 1차원 배열
print(np.arange(5,10)) #5 부터 1씩증가하는 1차원배열
print(np.zeros((2,2,))) # 영행렬 생성 
print(np.ones((2,3))) # 유닛 행렬
print(np.full((2,3),5)) # 모든 원소가 5인 행렬
print(np.eye(3)) #단위행렬


#################
# 차원 변환
#################
a = np.arange(20)
a = a.reshape((4,5))
print(b)





#2. 슬라이싱 인덱싱

a = [
    [1,2,3],
    [4,5,6],
    [7,8,9]
]
a = np.array(a)
a = a[0:2, 0:2]
a = a[1:, 1:]




# 3. 연산
# add, substract, multiply, divide
a = np.array([1,2,3])
b = np.array([4,5,6])

c = np.add(a,b)
c = np.multiply(a,b)
c = np.divide(a,b)
c = np.dot(a,b) # 행렬의 곱
s = np.sum(a) # 모든 원소의 합
a.sum(axis=0) # 열에 있는 원소의 합, 
a.mean()
a.std()
a.prod()



# 4. 난수 생성
x = np.random.uniform(size=100)
x.reshape(20,5)

s = np.random.normal(0,1,1000)

import matplotlib.pyplot as plt
plt.hist(s)