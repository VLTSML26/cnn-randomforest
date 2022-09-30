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
        _, self.img_rows, self.img_cols = x_train.shape
        self.num_classes = len(set(y_train))

        #NOTE: the following two lines must be put better
        self.y_train = keras.utils.to_categorical(y_train)
        self.y_test = keras.utils.to_categorical(y_test)

        self.input_shape = None
        self.check_data_format()

    def check_data_format(self):
        if keras.backend.image_data_format() == 'channels_first':
            self.x_train = np.expand_dims(self.x_train, axis=1)
            self.x_test = np.expand_dims(self.x_test, axis=1)
            self.input_shape = (1, self.img_rows, self.img_cols)
        else:
            self.x_train = np.expand_dims(self.x_train, axis=-1)
            self.x_test = np.expand_dims(self.x_test, axis=-1)
            self.input_shape = (self.img_rows, self.img_cols, 1)

def main():
    a = DataLoader()
    a.main()

if __name__ == '__main__':
    main()