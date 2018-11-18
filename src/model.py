from keras.layers import Input, Dense, Reshape, Flatten, Dropout, add
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam, SGD
from keras.models import Model, Sequential
from keras import activations
import cmath as math
from sklearn.model_selection import train_test_split
import random

from DataPreprocessor import DataPreprocessor

BATCH_SIZE=32
EPOCHS=2

def createModelSeverity():
    model = Sequential()
    model.add(Dense(32, activation='relu'))
    model.add(Dense(3, activation='softmax'))

    adam = Adam(lr=0.0001)
    model.compile(loss='categorical_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy'])

    return model

def createModelCasualties():
    model = Sequential()
    model.add(Dense(32, activation='relu'))
    model.add(Dense(39, activation='softmax'))

    adam = Adam(lr=0.0001)
    model.compile(loss='categorical_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy'])

    return model

def shuffleLists(a, b):
    combined = list(zip(a, b))
    random.shuffle(combined)
    return zip(*combined)

def trainSeverity(data):
    X, y = data.getDataForSeverity()
    shuffleLists(X, y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10)

    model = createModelSeverity()
    model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE)
    print(model.evaluate(X_test, y_test))

    model.save("models/severity.h5")

def trainCasualites(data):
    X, y = data.getDataForCasualties()
    shuffleLists(X, y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10)

    model = createModelCasualties()
    model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE)
    print(model.evaluate(X_test, y_test))

def main():
    dataprep = DataPreprocessor()
    dataprep.preprocess()

    trainCasualites(dataprep)

if __name__ == '__main__':
    main()
