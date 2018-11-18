import pandas as pd
import numpy as np
import math
from random import randint
from keras.utils import to_categorical
import itertools
import random

class DataPreprocessor():
    def __init__(self):
        self.file = pd.read_csv('dataset/data.csv')
        self.data = self.file.values

    def oneHotOne(self, item, max_bound):
        array = [0] * max_bound
        array[item] = 1

        return array

    def oneHot(self, x, exclude):
        maxes    = [1, 10, 12, 6, 23, 5, 6, 9, 8, 8, 38, 2]
        newArray = []

        #i indexes rows, j indexes colums in ith row
        for i in range(len(x)):
            newline = []
            for j in range(len(x[i])):
                if j in exclude:
                    newline.append(x[i][j])
                else:
                    newline = list(itertools.chain(newline,
                              self.oneHotOne(x[i][j], maxes[j]+1)))

            newArray.append(newline)

        return np.asarray(newArray)

    def setVehicleAgeBand(self, x):
        age = x[2]

        if age == -1:
            age = 4
        elif age >= 25:
            age = 12
        else:
            age = np.floor(age / 2)

        x[2] = int(age)

    def setTimeBand(self, x):
        time  = x[4]

        # If time is undefined
        if isinstance(time, float):
            hours = '12' # Assume midday
        else:
            hours = time.split(':')[0]

        x[4] = int(hours)

    
    def preprocess(self, data=[]):
        if (len(data) != 0):
            self.data = data
 
        for i in range(len(self.data)):
            self.setVehicleAgeBand(self.data[i])
            self.setTimeBand(self.data[i])

            self.fillMissingData(self.data[i])

        # Convert to np array & one hot
        self.data = np.asarray(self.data)
        print(self.data.max(axis=0))
        self.data = self.oneHot(self.data, [])
        print(self.data[0])

    def getDataForSeverity(self):
        X = self.data[:, :-42]
        y = self.data[:, -3:]
        return X, y

    def getDataForCasualties(self):

        X1 = self.data[:, :-42]
        X2 = self.data[:, -3:]
        X = np.column_stack((X1, X2))

        y = self.data[:, -42:-3]
        return X, y

    def fillMissingData(self, x):
        # Gender
        x[0] = x[0] - 1
        if x[0] != 0 and x[0] != 1:
            x[0] = randint(0,1)

        # Age band of driver
        x[1] = x[1] - 1
        if x[1] == -2:
            x[1] = randint(0,10)

        # Day of week
        x[3] = x[3] - 1
        if x[3] == -2:
            x[3] = randint(0, 6)

        # Road type
        if x[5] == 6:
            x[5] = 4
        elif x[5] == 7:
            x[5] = 5
        elif x[5] == 9:
            x[5] = 6
        elif x[5] == 12:
            x[5] = 7

        x[5] = x[5] - 1
        if x[5] == -2:
            x[5] = 6

        # Speed limit
        if x[6] == 0:
            x[6] = 40

        x[6] = int(x[6] / 10 - 1)

        # Junction detail
        if x[7] == -1:
            x[7] = 9

        # Weather conditions
        x[8] = x[8] - 1
        if x[8] == -2:
            x[8] = 8

        # Vehicle type
        elif x[9] == 1:
            x[9] = 0
        elif x[9] in [2, 3, 4, 5, 23, 97]:
            x[9] = 1
        elif x[9] in [8, 9, 90, -1]:
            x[9] = 2
        elif x[9] in [10, 11]:
            x[9] = 3
        elif x[9] == 19:
            x[9] = 4
        elif x[9] == 22:
            x[9] = 5
        elif x[9] in [20, 21, 98]:
            x[9] = 6
        elif x[9] in [16, 17]:
            x[9] = 7
        elif x[9] == 18:
            x[9] = 8
        else:
            x[9] = 2

        x[11] = x[11] - 1
