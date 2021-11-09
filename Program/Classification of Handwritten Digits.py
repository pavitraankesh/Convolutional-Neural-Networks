import tensorflow as tf
from tensorflow.keras import layers,models
from tensorflow import keras
import numpy as np

(X_train, y_train) , (X_test, y_test) = keras.datasets.mnist.load_data()

X_train = X_train / 255
X_test = X_test / 255
X_train = X_train.reshape(-1,28,28,1) 
X_test = X_test.reshape(-1,28,28,1)

cnn = models.Sequential([
    layers.Conv2D(filters=25, kernel_size=(3, 3), activation='relu', input_shape=(28,28,1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),    layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

cnn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
cnn.fit(X_train, y_train, epochs=10)

cnn.evaluate(X_test, y_test)

y_predicted_by_model = cnn.predict(X_test)
np.argmax(y_predicted_by_model[0])
