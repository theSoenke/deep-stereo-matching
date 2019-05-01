from keras.layers import Conv2D, BatchNormalization
from keras.models import Sequential


def create_model():
    model = Sequential()

    for i in range(9):
        model.add(Conv2D(64, 3, activation='relu'))
        model.add(BatchNormalization())

    return model
