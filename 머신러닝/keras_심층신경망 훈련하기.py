"""_summary_
1. 까다로운 그레이디언트 소실 또는 그레이디언트 폭주 문제에 직면할 수 있음. 아래쪽으로 갈수록 그레이디언트가 점점 작아지거나 커지는 현상임
 두 현상 모두 하위층을 훈련하기 매우 어렵게 만듭니다.
2. 데이터가 충분하지 않거나 레이블을 만드는 작업에 비용이 너무 많이 들 수 있음
3. 훈련이 극단적을 ㅗ느려질수 있음
4. 수백만개의 파라미터를 가진 모델은 훈련 세트에 과대적합될 위험이 큼



# 그레이디언트 소실과 폭줌 ㅜㄴ제

"""


import tensorflow as tf
from tensorflow import keras
# 케라스로 배치 정규화 구현하기

model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28,28]),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(300,activation="elu", kernel_initializer="he_normal", usr_bias=False),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(100,activation="elu", kernel_initializer="he_normal", usr_bias=False),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(10, activation="softmax")
])

model.summary()



import tensorflow as tf
from tensorflow import keras

fashion_mnist = keras.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test,y_test) = fashion_mnist.load_data()
X_train_full.shape
X_train_full.dtype

X_valid, X_train = X_train_full[:5000]/255.0 , X_train_full[5000:]/255.0
y_valid, y_train = y_train_full[:5000] , y_train_full[5000:]
X_test = X_test / 255.0




# 전이학습
model_A = model
model_B_on_A = keras.models.Sequential(model_A.layers[:-1])
model_B_on_A.summary()

# 해당 코드는 B_on_A 를 훈련할때 A에도 영향을 받습니다.
# 이를 원치 않는다면 클론으로 복제하면 됩니다.

model_A_clone = keras.models.clone_model(model_A)
model_A_clone.set_weights(model_A.get_weights())

# 여기서 문제는 새로운 출력층이 랜덤하게 초기화되어있으므로 큰 오차를 만들게 됩니다.(적어도 청므 몇 번의 에포크 동안)
# 이를 피하는 한 가지 방법은 처음 몇번의 에포크 동안 재사용된 층을 동결하고 새로운 층에게 적절한 가중치를 학습할 시간을 주는 것 입니다. 

for layer in model_B_on_A.layers[:-1]:
    layer.trainable=False
    
model_B_on_A.compile(loss="binary_crossentropy", optimizer="sgd", metrics=["accuracy"])

history = model_B_on_A.fit(X_train, y_train, epochs=4, validation_data=(X_valid, y_valid))

for layer in model_B_on_A.layers[:-1]:
    layer.trainable = True

optimizer = keras.optimizers.SGD(learning_rate=1e-4)
model_B_on_A.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])
history1 = model_B_on_A.fit(X_train, y_train, epochs=16, validation_data=(X_valid, y_valid))