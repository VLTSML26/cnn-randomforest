import tensorflow as tf
import numpy as np
from tensorflow import keras

class DataLoader():
    
    def __init__(
        self, 
        rawdata
    ):
        (x_train, y_train), (x_test, y_test) = rawdata.load_data()
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = keras.utils.to_categorical(y_train)
        self.y_test = keras.utils.to_categorical(y_test)

        self.num_classes = len(set(y_train))
        _, self.img_rows, self.img_cols = x_train.shape

        self.input_shape = None

def main():
    a = DataLoader()
    a.main()

if __name__ == '__main__':
    main()