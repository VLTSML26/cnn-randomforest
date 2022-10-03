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
    epochs = None
    dropout = None
    batch_size = None
    optimizer = None
    dataset = None

    def __init__(self, config_dict):
        """
        Constructor.
        
        Parameters
        ----------
        config_dict: dictionary for configuration of the neural network.
        """
        tf.random.set_seed(0)
        self.__dict__.update(config_dict)
        self.data = DataLoader(self.dataset)
        self.model = self.create_model()

    def get_optimizer(self):
        return self.optimizer

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
            # callbacks=[PrintEpoch()],
            verbose=0,
            validation_data=(self.data.x_test, self.data.y_test)
        )

def main():
    df = {
        'epochs': 5,
        'dropout': 0.2,
        'batch_size': 32,
        'optimizer': Nadam(),
        'dataset': keras.datasets.fashion_mnist,
    }
    aa = CNN(df)
    aa.fit()
    # print(aa.data.x_train.shape, aa.data.input_shape)
    
if __name__ == '__main__':
    main()