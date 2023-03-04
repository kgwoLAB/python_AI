import tensorflow as tf
from tensorflow import keras

fashion_mnist = keras.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test,y_test) = fashion_mnist.load_data()

X_train_full.shape
X_train_full.dtype

X_valid, X_train = X_train_full[:5000]/255.0 , X_train_full[5000:]/255.0
y_valid, y_train = y_train_full[:5000] , y_train_full[5000:]
X_test = X_test / 255.0

class_names = ["T-shirt/top","Trouser","Pullover","Dress","Coat","Sandal","Shirt","Sneaker","Bag","Ankle boot"]
class_names[y_train[0]]

# 시퀀셜 API로 순서대로 연결된 층을 일렬로 쌓아서 구성함.
model = keras.models.Sequential()
#Flatten 층은 입력 이미지를 1D 배녕로 반환합니다. reshape(-1,28*28) 과 같음
model.add(keras.layers.Flatten(input_shape=[28,28]))
model.add(keras.layers.Dense(300, activation="relu"))
model.add(keras.layers.Dense(100,activation="relu"))
model.add(keras.layers.Dense(10, activation="softmax"))





model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28,28]),
    keras.layers.Dense(300, activation="relu"),
    keras.layers.Dense(100,activation="relu"),
    keras.layers.Dense(10, activation="softmax"),
])

model.layers
model.summary()
hidden = model.layers[1]
hidden.name


# 층의 모든 파라미터는 get_widgets() 메서드와 set_widgets() 메서드를 사용해 접근할수 있습니다.
weights, biases = hidden.get_weights()
weights
biases
# Dense 층은 연결 가중치를 무작위로 초기화 합니다. 편향은 0으로 초기화하는대 다르게 설정할수도 있읍니다.

# compile() 메서드를 호출하여 
# 사용할 손실함수, 옵티마이저, 부가적으로 계산할 지표를 추가로 지정할 수 있습니다. 

model.compile(loss="sparse_categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])
"""
레이블이 정수이며 클래스가 베타적 : sparse_categorical_crossentropy
샘플마다 클래스별 타깃 확률을 가지고 있다면 : categorical_crossentropy
이진분류나 다중 레이블 이진 분류 수행 한다면 sigmoid : binary_crossentropy


옵티마이저에 sgd를 입력하면 sgd를 사용해 모델을 훈련한다는 의미입니다.
다른말로는 역전파 알고리즘을 수행함.

마지막으로 분류기이기에 정확도를 측정하기위해 accuracy로 지정합니다.
"""
history = model.fit(X_train, y_train, epochs=30, validation_data=(X_valid,y_valid))


# 매개변수의 편중을 처리하기위해 class_weights 매개변수 지정가능

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
pd.DataFrame(history.history).plot(figsize=(8,5))
plt.grid(True)
plt.gca().set_ylim(0,1) # 수직축의 범위를 0~1로 설정합니다.
plt.show()


# 일반화 오차 검사
model.evaluate(X_test, y_test)


X_new = X_test[:3]
y_proba = model.predict(X_new)
y_proba.round(2)

# 가장 높은 확률을 가진 클래스에만 관심을 가진다면
y_pred = model.predict_classes(X_new)
y_pred
np.array(class_names)[y_pred]

y_new = y_test[:3]
np.array(class_names)[y_new]



# 시퀀셜 API를 사용하여 회귀용 다층 퍼셉트론 만들기
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

housing = fetch_california_housing()

X_train_full, X_test, y_train_full, y_test = train_test_split(housing.data, housing.target)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)

# 출력층이 활성화 함수가 없는 하나의 뉴런을 가지며 손실함수로 평균 제곱 오차를 사용한다는 점이 차이점
# 데이터 셋에는 잡음이 많으면 과대 적합을 막기위해 뉴런수가 적은 은닉층 하나만 사용

model = keras.models.Sequential([
    keras.layers.Dense(30, activation="relu", input_shape=X_train.shape[1:]),
    keras.layers.Dense(1)
])

model.compile(loss="mean_squared_error", optimizer="sgd")
history = model.fit(X_train,y_train, epochs=20, validation_data=(X_valid,y_valid))
mse_test = model.evaluate(X_test, y_test)
X_new = X_test[:3]
y_pred = model.predict(X_new)



###############################################################################################################
# 케라스는 함수형 API를 제공합니다.
# 모델의 입력을 정의하며, 한 모델은 여러 개의 입력을 가질 수 있음.
input_ = keras.layers.Input(shape=X_train.shape[1:])
# 30개의 뉴런과ㅓ RELU 활성화 함수를 가진 Dense 층을 만들며 입력과함께 함수처럼 호출됩니다. 
hidden1 = keras.layers.Dense(30, activation="relu")(input_)
hidden2 = keras.layers.Dense(30, activation="relu")(hidden1)
# Concatenate 층을 만들어 함수처럼 호출하여 두 번째 은닉층의 출력과 입력을 연결합니다. 주어진입력을 바로 사용해 호출함.
concat = keras.layers.Concatenate()([input_, hidden2])
output = keras.layers.Dense(1)(concat)
#모델을 만듬
model = keras.Model(inputs=[input_], outputs=[output])
###############################################################################################################
# 모델을 컴파일한다음 훈련, 평가, 예측을 수행할 수 있음. 
# 일부 특성을 짧은 경로로 전달하고 다른 특성들은 깊은 경로로 전달하고 싶다면, 이럴 경우는 여러 입력을 사용하는 것 입니다. 
###############################################################################################################





###############################################################################################################
# 여러 개의 입력 다루기
input_A = keras.layers.Input(shape=[5], name="wide_input")
input_B = keras.layers.Input(shape=[6], name="deep_input")
hidden1 = keras.layers.Dense(30, activation="relu")(input_B)
hidden2 = keras.layers.Dense(30, activation="relu")(hidden1)
concat = keras.layers.concatenate([input_A, hidden2])
output = keras.layers.Dense(1, name="output")(concat)
model = keras.Model(inputs=[input_A, input_B], outputs=[output])

model.summary()

model.compile(loss="mse", optimizer=keras.optimizers.SGD(learning_rate=1e-3))


X_train_A, X_train_B = X_train[:,:5], X_train[:, 2:]
X_valid_A, X_valid_B = X_valid[:,:5], X_valid[:,2:]
X_test_A, X_test_B = X_test[:, :5], X_test[:,2:]
X_new_A, X_new_B = X_test_A[:3], X_test_B[:3]

history = model.fit((X_train_A, X_train_B), y_train, epochs=20,
                    validation_data=((X_valid_A, X_valid_B), y_valid))




# 보조 출력
input_A = keras.layers.Input(shape=[5], name="wide_input")
input_B = keras.layers.Input(shape=[6], name="deep_input")
hidden1 = keras.layers.Dense(30, activation="relu")(input_B)
hidden2 = keras.layers.Dense(30, activation="relu")(hidden1)
concat = keras.layers.concatenate([input_A, hidden2])
output = keras.layers.Dense(1, name="main_output")(concat)
aux_output = keras.layers.Dense(1, name="aux_output")(hidden2)
model = keras.Model(inputs=[input_A, input_B], outputs=[output, aux_output])

model.compile(loss=["mse","mse"], loss_weights=[0.9,0.1], optimizer="sgd")
history = model.fit([X_train_A, X_train_B],[y_train,y_train] , epochs=20,
                    validation_data=([X_valid_A, X_valid_B], [y_valid,y_valid]))
# 모델을 평가하면 케라스는 개별 손실과 함께 총 손실을 반환합니다.
total_loss, main_loss, aux_loos = model.evaluate([X_test_A,X_test_B], [y_test, y_test])
y_pred_main, y_pred_aux = model.predict([X_new_A,X_new_B])




"""
서브클래싱 API로 동적 모델 만들기.
시퀀셜 API와 함수형 API는 선억적임.
1. 사용할 층과 연결 방식을 먼저정의
2. 모델에 데이터를 주입하여 훈련이나 추론을 시작할수 있음
3. 모델 저장, 복사 및 공유가 쉬움
4. 또한 모델의 구조를  출력하거나 분석하기 좋음
5. 프레임워크가 크기를 짐작하고 타입을 확인하여 에러를 일찍 발견할 수 있음.

6. 단점은 반복문을 퐘하고 다양한 크기를 다루어야 하며 조건문을 가지는 등 여러 가지 동적인 구조를 필요로함.
7. 명령형 프로그래밍 스타일이라면 서브 클래싱 API가 정답입니다.     
"""
class WideAndDeepModel(keras.Model):
    def __init__(self, units=30, activation="relu", **kwargs):
        super().__init__(**kwargs)
        self.hidden1 = keras.layers.Dense(units, activation=activation)
        self.hidden2 = keras.layers.Dense(units, activation=activation)
        self.main_output = keras.layers.Dense(1)
        self.aux_output = keras.layers.Dense(1)
    
    def call(self, inputs):
        input_A, input_B = inputs
        hidden1 = self.hidden1(input_B)
        hidden2 = self.hidden2(hidden1)
        concat = keras.layers.concatenate([input_A, hidden2])
        main_output = self.main_output(concat)
        aux_output = self.aux_output(hidden2)
        return main_output, aux_output

model = WideAndDeepModel()

# call 메서드의 input 매개변수를 사용하여 계산가능,
# 해당 구문 안에서는 for문 if문 저수준 연산을 사용할수있습니다.
# 유연성이 높아지면 그에 따른 비용이 발생합니다.
# 모델 구조가 call 메서드 안에 숨겨져 있기에 케라스가 이를 쉽게 분석ㅎ랄 수 없어 모델을 저장하거나 복사할수 없ㅅ브니다.






# 모델 저장과 복원
model.save("my_keras_model.h5")
# HDF5 포맷을 사용하여 모델 구조와 층의 모든 모델 파라미터를 저장합니다. 또한 현재 상태를 포함한 옵티마이저도 저장합니다.
model = keras.models.load_model("my_keras_model.h5")

# 시퀀셜 , 함수형 API는 해당방식을쓰지만
# 모델 서브클래싱에는 사용할수 없습니다. save_weights() 와 load_weights() 메서드를 사용해 모델 파라미터를 저장 및 복원할 수 있습니다.

# 대규모 데이터셋을 훈련하다 중간에 문제가 생기는 가능성을 방지하기위해 체크포인트를 저장해야합니다.
# 바로 체크포인트




# 콜백 사용하기
checkpoint_cb = keras.callbacks.ModelCheckpoint("my_keras_model.h5")
history = model.fit([X_train_A, X_train_B],y_train , epochs=10, validation_data=([X_valid_A, X_valid_B], [y_valid,y_valid]), callbacks=[checkpoint_cb])

checkpoint_cb = keras.callbacks.ModelCheckpoint("my_keras_model.h5", save_best_only=True)
history = model.fit([X_train_A, X_train_B],y_train , epochs=10, validation_data=([X_valid_A, X_valid_B], [y_valid,y_valid]), callbacks=[checkpoint_cb])
model = keras.models.load_model("my_keras_model.h5") # 최상의 모델로 복원하기.

# 조기 종료를 구현하는 방법은 다른수도 있습니다.
early_stopping_cb = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
history = model.fit([X_train_A, X_train_B],y_train , epochs=10, validation_data=([X_valid_A, X_valid_B], [y_valid,y_valid]), callbacks=[checkpoint_cb,early_stopping_cb])
# 모델이 향상되지 않으면 훈련이 자동으로 중지되기에 에포크 숫자를 크게 지정해도 됩니다. 
# 해당 경우는 최상의 가중치로 자동으로 복원되기에 따로 건들필요가 없다.


# 훈련손실과 검증 손실을 출력해 과대적합을 감지합니다.
# 사용자 정의 콜백 만들기
class PrintValTrainRatioCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch,logs):
        print("\nval/train: {:.2f}".format(logs["val_loss"] / logs["loss"]))
        
        
# p303을 통해 분석을 찾을수 있습니다.

# 텐서보드를 사용해 시각화 하기
import os
root_logdir = os.path.join(os.curdir, "my_logs")
print(root_logdir)

def get_run_logdir():
    import time
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)

run_logdir = get_run_logdir()

tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)
history = model.fit([X_train_A, X_train_B],y_train , epochs=10, validation_data=([X_valid_A, X_valid_B], [y_valid,y_valid]), callbacks=[tensorboard_cb])

# Tensorboard() 콜백이 로그 디렉토리를 생성하며 훈련하는 동안 이벤트 파일을 만들고 서머리를 기록합니다.

# tensorboard --logdir=./my_logs --port=6006
#load_ext tensorboard
#tensorboard --logdir=./my_logs --port=6006


test_logdir = get_run_logdir()
writer = tf.summary.create_file_writer(test_logdir)

# 케라스 모델을 사이킷런 추정기처럼 보이도록 바꾸어야 합니다.
# 먼저 일련의 하이퍼파라미터로 케라스 모델을 만들고 컴파일하는 함수를 만듭니다. 

def build_model(n_hidden=1, n_neurons=20, learning_rate=3e-3, input_shape=[8]):
    model =keras.models.Sequential()
    model.add(keras.layers.InputLayer(input_shape=input_shape))
    for layer in range(n_hidden):
        model.add(keras.layers.Dense(n_neurons, activation="relu"))
    model.add(keras.layers.Dense(1))
    optimizer = keras.optimizers.SGD(learning_rate=learning_rate)
    model.compile(loss="mse", optimizer = optimizer)
    return model


# 입력크기와 은닉층 개수, 뉴런 개수로 단변량 회귀를 위한 모델을 만들어 ?

keras_reg = keras.wrappers.scikit_learn.KerasRegressor(build_model)

# 케라스 모델을 감싸는 간단한 래?퍼? 일반적인 사이킷런 회귀 추정기 처럼 객체를 사용할수 있습니다.
# fit score predict
keras_reg.fit(X_train,y_train,epochs=100, validation_data=(X_valid,y_valid), callbacks=[keras.callbacks.EarlyStopping(patience=10)])
mse_test =keras_reg.score(X_test, y_test)
y_pred = keras_reg.predict(X_new)




# 사이킷런은 손실이아닌 점수를 계산하기에 출력 점수는 음수의 MSE 입니다. 
from scipy.stats import reciprocal
from sklearn.model_selection import RandomizedSearchCV

param_distribs={
    "n_hidden":[0,1,2,3],
    "n_neurons":np.arange(1,100),
    "learning_rate":reciprocal(3e-4, 3e-2),
}

rnd_search_cv = RandomizedSearchCV(keras_reg, param_distribs, n_iter=10, cv=3)
rnd_search_cv.fit(X_train,y_train,epochs=100, validation_data=(X_valid,y_valid), callbacks=[keras.callbacks.EarlyStopping(patience=10)])



# 왜 안되누 ;;; - Kernel Crashed 

# 하이퍼파라미터 초지ㅓㄱ화에 사용할 수 있는 파이썬 라이브러리 정렬함
"""_summary_
Hyperopt - 모든 종류의 복잡한 탐색 공간에 대해 최적화를 수행할 수 있는 잘 알려진 라이브러리 입니다.
Hyperas, kopt, Talos : 케라스 모델을 위한 하이퍼파라미터 최적화 라이브러리 입니다.
케라스 튜너 : 사용하기 쉬운 케라스 하이퍼파라미터 최적화 라이브러리 이며 구글이 만들어서 시각화분석을 포함한 클루으드 서버시
scikit-optimizer 범용 최적화 라이브러리 입니다. 

Spearmint 베이즈 최적화
Hyperband : 빠른 하이퍼파라미터 튜닝 라이브러리 입니다.
Sklearn-Deap : 진화 알고리즘 기반의 하이퍼파라미터 최적화 라이브러리 입니다. 
"""

"""
하이퍼파라미터 

# 은닉층 개수
하나로 시작해도 괜찮음
복잡한 문제는 심층 신경망이 파라미터 효율성이 더 좋음 
전이학습
일반적으로 층의 뉴런 수보다 층 수를 늘리는 쪽이 이득이 많습니다.

## 학습률, 배치 크기, 다른 하이퍼 파라미터
# 학습률
일반적으로 최적읙습률은 최대 학습률의 절반정도
좋은 학습률을 찾는 한가지 방법은 매우 낮은 학습률에서 시작해 점진적으로 매우 큰 학습률까지 수백번 반복하며 모델을 훈련합니다.
반복마다 일정한 값을 학습률에 곱합니다. exp(log(10^6)/500)를 500번 반복하여 곱합니다. 
로그 스케일로 조정된 학습률을 사용하여 학습률에 대한 손실을 그래프로 그리면 처음에 손실이 줄어드는 것이보입니다.
하지만 학습률이 커지게되면 손실이 다시 커지게 됩니다. 
최적의 학습률은 손실이 다시 상승하는 지점보다 조금 아래에 있습니다. (일반적으로 상승점보다 약 10배 낮은 지점입니다.)

# 옵티마이저
고전적인 평범한 미니배치 경사 하강법보다 더좋은 옵티마이저 들이 많습니다.

# 배치 크기
큰 배치 크기를 사용하는 것의 주요 장점은 GpU와 같은 하드웨어 가속기를 효율적으로 활용할 수 있다는 점입니다.
따라서 훈련 알고리즘이 초당 더 많은 샘플을 처리할 수 있습니다.
많은 연구자들과 기술자들은 GPU RAM에 맞는 가장 큰 배치 크기를 사용하라 권하지만
실전에서 큰 배치를 사용하면 훈련 초기에 종종 불안정하게 훈련됩니다. 
일반화 성능이 현저히 낮아질수 있습니다. 

# 활성화 함수
일반적으로 RelU 함수가 모든 은닉층에 좋은 기본값입니다.

# 반복횟수
대부분 설정할 필요 없이 조기 종료를 선호 합니다. 



"""





















"""




