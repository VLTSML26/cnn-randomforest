import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging, os

from abc import ABC, abstractmethod
from tensorflow import keras
from keras.optimizers import SGD, Adam, RMSprop, Adagrad, Adadelta, Adamax, Nadam
from keras.layers import Flatten, Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout
from keras.models import Sequential
from sklearn import ensemble
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from data_loader import DataLoader

logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.random.set_seed(0)
np.random.seed(0)

class Classifier(ABC):
    """
    Virtual class equipped with the principal abstract methods to be used in a
    classification problem.
    """

    def __init__(
        self,
        dataset,
        pca_percent
    ):
        """
        Virtual class constructor.

        Parameters
        ----------
        dataset: for now accepts only keras.datasets object containing data from
        the fashion_mnist since this is the task of the exercise. This can easily
        be extended in further applications.
        
        Note
        ----
        The order of the virtual methods called in this constructor matter!
        """
        self.data = DataLoader(dataset)
        self.pca_percent = pca_percent
        self.reshape_data()
        if self.pca_percent is not None:
            self.reduce_dimensions()
        self.model = self.create_model()

    @abstractmethod
    def reshape_data(self):
        """
        Reshapes dataset according to algorithm used for classification.
        """
        pass

    @abstractmethod
    def reduce_dimensions(self):
        """
        Reduce dimensions if needed.
        """
        pass

    @abstractmethod
    def create_model(self):
        """
        Creates model.
        """
        pass

    @abstractmethod
    def train(self):
        """
        Model training method.
        """
        pass

    @abstractmethod
    def evaluate(self):
        """
        Evaluates performance on validation data.
        """
        pass

    def summary(self):
        try:
            return self.model.summary()
        except:
            pass

class CNN(Classifier):
    """
    Convolutional Neural Network (CNN) from parent class Classifier.
    """
    epochs = None
    dropout = None
    batch_size = None
    optimizer = None

    def __init__(self, dataset, config_dict, pca_percent=None):
        """
        CNN constructor.
        
        Parameters
        ----------
        dataset: see virtual class constructor.
        
        config_dict: dictionary containing the neural network parameters and the
        dataset as well.
        """
        self.__dict__.update(config_dict)
        super().__init__(dataset, pca_percent)
        
    def reshape_data(self):
        self.data.y_train = keras.utils.to_categorical(self.data.y_train)
        self.data.y_test = keras.utils.to_categorical(self.data.y_test)
        if keras.backend.image_data_format() == 'channels_first':
            self.data.x_train = np.expand_dims(self.data.x_train, axis=1)
            self.data.x_test = np.expand_dims(self.data.x_test, axis=1)
            self.data.input_shape = (1, self.data.img_rows, self.data.img_cols)
        else:
            self.data.x_train = np.expand_dims(self.data.x_train, axis=-1)
            self.data.x_test = np.expand_dims(self.data.x_test, axis=-1)
            self.data.input_shape = (self.data.img_rows, self.data.img_cols, 1)

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

    def evaluate(self):
        """
        Evaluate the performance of the CNN for the classification problem.

        Returns
        -------
        (score, cm, history): tuple
            score: dict containing infos about the NN optimizer and its accuracy
            cm: np.array containing the confusion matrix
            history: keras.callbacks.History object returned by the fit method
        """
        history = self.train()
        y_pred = np.argmax(self.model.predict(self.data.x_test), axis=1)
        rounded_labels = np.argmax(self.data.y_test, axis=1)
        cm = confusion_matrix(rounded_labels, y_pred)
        loss, acc = self.model.evaluate(
            self.data.x_test,
            self.data.y_test,
            verbose=0
        )
        score = {
            'Optimizer': self.optimizer,
            'Loss': '%.4f'%loss,
            'Accuracy': '%.4f'%acc,
        }
        return score, cm, history

class RandomForest(Classifier):
    n_estimators = None
    criterion = None
    max_samples = None

    def __init__(self, dataset, config_dict, pca_percent=None):
        self.__dict__.update(config_dict)
        super().__init__(dataset, pca_percent)
    
    def reshape_data(self):
        self.data.x_train = self.data.x_train.reshape(
            len(self.data.x_train),
            self.data.img_cols * self.data.img_rows
        )
        self.data.x_test = self.data.x_test.reshape(
            len(self.data.x_test),
            self.data.img_cols * self.data.img_rows
        )
        if self.pca_percent is not None:
            self.rescale_data()

    def rescale_data(self):
        self.data.x_train = StandardScaler().fit_transform(self.data.x_train)
        self.data.x_test = StandardScaler().fit_transform(self.data.x_test)

    def reduce_dimensions(self):
        pca = PCA(n_components=self.compute_pcacomponents())
        pca.fit(self.data.x_train)
        self.data.x_train = pca.transform(self.data.x_train)
        self.data.x_test = pca.transform(self.data.x_test)

    def compute_pcacomponents(self):
        covmat = np.cov(self.data.x_train.T)
        eval, _ = np.linalg.eig(covmat)
        eval_percent = [this/sum(eval) for this in sorted(eval, reverse=True)]
        variance_contributions = np.cumsum(eval_percent)
        return len(variance_contributions) - sum(variance_contributions > self.pca_percent)

    def create_model(self):
        return ensemble.RandomForestClassifier(
            self.n_estimators,
            criterion=self.criterion,
            max_samples=self.max_samples,
            random_state=0,
            n_jobs=-1,
        )
    
    def train(self):
        self.model.fit(self.data.x_train, self.data.y_train)

    def evaluate(self):
        """
        Evaluate the performance of the random forest for the classification problem.

        Returns
        -------
        (acc, cm, report): tuple
            acc: float between 0 and 1
            cm: np.array containing the confusion matrix
            report: 
        """
        self.train()
        y_pred = self.model.predict(self.data.x_test)
        cm = confusion_matrix(self.data.y_test, y_pred)
        acc = accuracy_score(self.data.y_test, y_pred)
        report = classification_report(self.data.y_test, y_pred, output_dict=True)
        return cm, acc, report


"""
WHAT FOLLOWS IS USED FOR TESTING
"""
def main():
    # df = {
    #     'epochs': 5,
    #     'dropout': 0.2,
    #     'batch_size': 32,
    #     'optimizer': 'SGD',
    # }
    # aa = CNN(keras.datasets.fashion_mnist, df)
    # score, cm, history = aa.evaluate()
    df2 = {
        'n_estimators': 100,
        'criterion': 'entropy',
        'max_samples': 0.5
    }
    bb = RandomForest(keras.datasets.fashion_mnist, df2, pca_percent=0.9)
    cm, acc, report = bb.evaluate()
    print(type(report))
    print(acc)
    print(type(cm))
    

if __name__ == '__main__':
    main()