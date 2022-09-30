import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.optimizers import SGD, Adam, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam
from keras.layers import Flatten, Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout
from keras.models import Sequential

from data_loader import DataLoader

class CNN(object):

    def __init__(
        self,
        dataset,
        dropout = 0.2,
        epochs = 5,
        batch_size = 32,
        optimizer = Nadam()
    ):
        tf.random.set_seed(0)
        self.data = DataLoader(dataset)
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.model = self.create_model()

    def create_model(self):
        """
        Create CNN with the following specs:
        """
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

    def summary(self):
        return self.model.summary()

    def fit(self):
        return self.model.fit(
            self.data.x_train,
            self.data.y_train,
            batch_size=self.batch_size,
            epochs=self.epochs,
            callbacks=[PrintEpoch()],
            verbose=0,
            validation_data=(self.data.x_test, self.data.y_test)
        )

class PrintEpoch(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(".", end="")
    def on_train_end(self, logs=None):
        print()

def main():
    aa = CNN(keras.datasets.fashion_mnist)
    aa.fit()
    # print(aa.data.x_train.shape, aa.data.input_shape)
    
if __name__ == '__main__':
    main()