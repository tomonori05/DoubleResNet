import numpy as np
from abc import abstractmethod

from keras.datasets import cifar10
from keras.datasets import cifar100
from keras.datasets import mnist
import keras.utils.np_utils as kutils

class abstract_dataset:

    @abstractmethod
    def download(self):
        raise NotImplementedError()
    
    @abstractmethod
    def get_shape(self):
        raise NotImplementedError()
    @abstractmethod
    def get_name(self):
        raise NotImplementedError()
    @abstractmethod
    def get_categorical(self):
        raise NotImplementedError()

    def __init__(self,is_zero_center=False):
        (self.raw_trainX, trainY), (self.raw_testX, testY) = self.download()

        trainX = self.raw_trainX.astype('float32')
        trainX /= 255
        testX = self.raw_testX.astype('float32')
        testX /=255

        trainY = kutils.to_categorical(trainY)
        testY = kutils.to_categorical(testY)

        if is_zero_center == True:
            trainX -= 0.5
            testX -= 0.5

        self.trainX = trainX
        self.testX = testX
        self.trainY = trainY
        self.testY = testY

        #self.init_shape = (3, 32, 32) if K.image_dim_ordering() == 'th' else (32, 32, 3)
        #self.init_shape = (3,32,32)
        self.init_shape = self.get_shape()

class cifer10_datasets(abstract_dataset):

    def download(self):
        return cifar10.load_data()

    def get_shape(self):
        return (3, 32, 32)

    def get_name(self):
        return 'cifer10'
    
    def get_categorical(self):
        return 10

class cifer100_datasets(abstract_dataset):
    
    def download(self):
        return cifar100.load_data(label_mode='fine')

    def get_shape(self):
        return (3, 32, 32)

    def get_name(self):
        return 'cifer100'
    
    def get_categorical(self):
        return 100

class mnist_dataset(abstract_dataset):

    def download(self):
        (trainX, trainY), (testX, testY) = mnist.load_data()
        trainX_size = trainX.shape[0]
        testX_size = testX.shape[0]
        trainX = np.array(trainX)
        testX = np.array(testX)
        
        trainX = np.reshape(trainX,[trainX_size,28,28,1])
        testX = np.reshape(testX,[testX_size,28,28,1])
        
        return (trainX, trainY), (testX, testY)

    def get_shape(self):
        return (1, 28, 28)

    def get_name(self):
        return 'mnist'
    
    def get_categorical(self):
        return 10


