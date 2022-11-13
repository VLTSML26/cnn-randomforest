import tensorflow as tf
import numpy as np
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

class DataLoader():
    
    def __init__(
        self, 
        rawdata,
        pca_percent
    ):

        # Data loading
        (x_train, self.y_train), (x_test, self.y_test) = rawdata.load_data()
        self.x_train = x_train
        self.x_test = x_test
        self.num_classes = len(set(self.y_train))
        _, self.img_rows, self.img_cols = x_train.shape
        self.input_shape = None
        self.reshape()
        self.rescale()

        # PCA dimension reduction
        self.pca_percent = pca_percent
        if self.pca_percent is not None:
            self.reduce_dimensions()
        
    def reduce_dimensions(self):
        pca = PCA(n_components=self.compute_pcacomponents())
        pca.fit(self.x_train)
        self.x_train = pca.transform(self.x_train)
        self.x_test = pca.transform(self.x_test)

    def reshape(self):
        self.x_train = self.x_train.reshape(len(self.x_train), -1)
        self.x_test = self.x_test.reshape(len(self.x_test), -1)
    
    def rescale(self):
        self.x_train = StandardScaler().fit_transform(self.x_train)
        self.x_test = StandardScaler().fit_transform(self.x_test)

    def compute_pcacomponents(self):
        covmat = np.cov(self.x_train.T)
        eval, _ = np.linalg.eig(covmat)
        eval_percent = [this/sum(eval) for this in sorted(eval, reverse=True)]
        variance_contributions = np.cumsum(eval_percent)
        n_comps = len(variance_contributions) - sum(variance_contributions > self.pca_percent)
        sqrt_comp = int(np.ceil(np.sqrt(n_comps)))
        self.img_cols = sqrt_comp
        self.img_rows = sqrt_comp
        return sqrt_comp**2

def main():
    a = DataLoader()
    a.main()

if __name__ == '__main__':
    main()