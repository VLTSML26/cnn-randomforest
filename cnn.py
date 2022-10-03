from abc import ABC
from random import Random
import tensorflow as tf
import pandas as pd
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.optimizers import SGD, Adam, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam
from keras.layers import Flatten, Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout
from keras.models import Sequential
from sklearn import ensemble

from data_loader import DataLoader

class Classifier:

    def __init__(self, dataset):
        self.data = DataLoader(dataset)
        self.model = self.create_model()

    def create_model(self):
        pass

    def train(self):
        pass

    def evaluate(self):
        try:
            return self.model.evaluate(self.data.x_test, self.data.y_test)
        except:
            return self.model.score(
                self.data.x_test.reshape(len(self.data.x_test), self.data.img_cols * self.data.img_rows),
                self.data.y_test
            )

    def summary(self):
        try:
            return self.model.summary()
        except:
            pass

    # def table_evaluation(self):
    #     evaluate = self.evaluate()
    #     summary = {
    #         'Loss': evaluate[0],
    #         'Accuracy': evaluate[1],
    #     }
    #     df = pd.DataFrame.from_dict(summary)
    #     df = df.append({
    #         'Loss': tmp[0],
    #         'Accuracy': tmp[1],
    #     })

    #     yield df

class CNN(Classifier):
    epochs = None
    dropout = None
    batch_size = None
    optimizer = None

    def __init__(self, dataset, config_dict):
        self.__dict__.update(config_dict)
        super().__init__(dataset)
        
    def create_model(self):
        model = Sequential()
        model.add(
            Conv2D(
                self.data.num_classes,
                kernel_size=(5, 5),
                activation='relu',
                input_shape=self.data.input_shape,
            )
        )
        model.add(MaxPooling2D())
        model.add(Dropout(0.2))
        model.add(
            Conv2D(
                self.data.num_classes,
                kernel_size=(3, 3),
                activation='relu'
            )
        )
        model.add(MaxPooling2D())
        model.add(Dropout(0.2))
        model.add(Flatten())
        model.add(Dense(self.data.num_classes, activation='softmax'))
        
        model.compile(
            loss=keras.losses.categorical_crossentropy,
            optimizer=self.optimizer,
            metrics=['acc']
        )
        return model

    def train(self, verbose: int = 0):
        return self.model.fit(
            self.data.x_train,
            self.data.y_train,
            batch_size=self.batch_size,
            epochs=self.epochs,
            verbose=verbose,
            validation_data=(self.data.x_test, self.data.y_test)
        )

class RandomForest(Classifier):
    n_estimators = None

    def __init__(self, dataset, config_dict):
        self.__dict__.update(config_dict)
        super().__init__(dataset)
        
    def create_model(self):
        return ensemble.RandomForestClassifier(self.n_estimators)
    
    # TODO: the constructor should reshape data, not the method. Also might be
    # better if DataLoader could pass already well-shaped data by accessing info
    # about the class calling it
    def train(self):
        self.model.fit(
            self.data.x_train.reshape(len(self.data.x_train), self.data.img_cols * self.data.img_rows),
            self.data.y_train
        )

def main():
    df = {
        'epochs': 5,
        'dropout': 0.2,
        'batch_size': 32,
        'optimizer': 'SGD',
    }
    aa = CNN(keras.datasets.fashion_mnist, df)
    aa.train(verbose=1)
    print(aa.evaluate())
    df2 = {
        'n_estimators': 100
    }
    bb = RandomForest(keras.datasets.fashion_mnist, df2)
    bb.train()
    print(bb.evaluate())

if __name__ == '__main__':
    main()