from keras.layers import Input, Dense, Reshape, Flatten, Dropout, add
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam, SGD
from keras.models import Model, Sequential
from keras import activations

from DataPreprocessor import DataPreprocessor

DATA_PATH = 'data/data.csv'

def createModel():
    pass

def main():
    dataprep = DataPreprocessor()
    dataprep.preprocess()

if __name__ == '__main__':
    main()
