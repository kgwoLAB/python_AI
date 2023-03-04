import tensorflow as tf
from tensorflow import keras

fashion_mnist = keras.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test,y_test) = fashion_mnist.load_data()
X_train_full.shape
X_train_full.dtype

X_valid, X_train = X_train_full[:5000]/255.0 , X_train_full[5000:]/255.0
y_valid, y_train = y_train_full[:5000] , y_train_full[5000:]
X_test = X_test / 255.0




import matplotlib.pyplot as plt

n_rows = 4
n_cols = 10
plt.figure(figsize=(n_cols * 1.2, n_rows * 1.2))
for row in range(n_rows):
    for col in range(n_cols):
        index = n_cols * row + col
        plt.subplot(n_rows, n_cols, index + 1)
        plt.imshow(X_train[index], cmap="binary", interpolation="nearest")
        plt.axis('off')
        plt.title(y_train[index], fontsize=12)
plt.subplots_adjust(wspace=0.2, hspace=0.5)
plt.show()




class_names = ["T-shirt/top","Trouser","Pullover","Dress","Coat","Sandal","Shirt","Sneaker","Bag","Ankle boot"]

model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28,28]),
    keras.layers.Dense(300, activation="relu"),
    keras.layers.Dense(100, activation="relu"),
    keras.layers.Dense(10, activation="softmax")  
])

import os
root_logdir = os.path.join(os.curdir, "my_logs")
print(root_logdir)

def get_run_logdir():
    import time
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)

run_logdir = get_run_logdir()
tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)

checkpoint_cb = keras.callbacks.ModelCheckpoint("my_keras_model.h5", save_best_only=True)

model.compile(loss="sparse_categorical_crossentropy", optimizer="sgd", metrics=["accuracy"],)
model.summary()
history = model.fit(X_train,y_train, epochs=30, validation_data=(X_valid, y_valid),callbacks=[tensorboard_cb, checkpoint_cb])
print(history)



# 심층 다층 퍼셉트론
# 최적의 학습률을 찾아라
# 학습률을 지속적으로 증가시키면서 손실을 그래프로 그립니다.
# 손실이 다시 증가하는 지점을 찾습니다. 
# 체크포인트 저장
# 조기 종료
# 텐서보드로 학습곡선 그리기 



