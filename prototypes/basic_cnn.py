# https://towardsdatascience.com/building-a-convolutional-neural-network-cnn-in-keras-329fbbadc5f5

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten # https://keras.io/layers/core/
from keras.utils import to_categorical

# MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# PREPARE DATA
# Convert to 28x28 images in greyscale
X_train = X_train.reshape(60000,28,28,1)
X_test  = X_test.reshape(10000,28,28,1)

# Convert int-vectors to binary matrices
y_train = to_categorical(y_train)
y_test  = to_categorical(y_test)

# CREATE MODEL
# kernel_size=3       -> Size of filter matrix [3,3]
# activation='relu'   -> ReLu activation https://user.phil.hhu.de/~petersen/SoSe17_Teamprojekt/AR/neuronalenetze.html#aktivierungsfunktionen
model = Sequential()
model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(28,28,1))) # 1. Layer
model.add(Conv2D(64, kernel_size=3, activation='relu'))                        # 2. Layer
model.add(Flatten())                                                           # Flatten input
model.add(Dense(10, activation='softmax'))                                     # 3. Layer

# metrics=['accuracy'] -> measure model performance by accuracy
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# TRAIN
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3)

# TEST
model.predict(X_test[:4])