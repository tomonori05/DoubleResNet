import numpy as np
import os
import json
import datetime
import time

import keras.utils.np_utils as kutils
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau, TensorBoard
#from ModelBuilder import ResnetBuilder
from model_module import ModelBuilder
from keras import backend as K
from datetime import datetime
from keras import optimizers,losses

#from json_writer import json_write
from tools import json_writer

class ResNetTester:

    def setDataset(self,dataset):
        self.trainX = dataset.trainX
        self.trainY = dataset.trainY
        self.testX = dataset.testX
        self.testY = dataset.testY
        
        test_data_shape = np.shape(self.testX)
        train_data_shape = np.shape(self.trainX)
        self.dataset_name = dataset.get_name()
        self.test_data = test_data_shape[0]
        self.train_data = train_data_shape[0]

        #バッチサイズなど
        self.batch_size = 128
        self.nb_epoch = 40
        self.validation_split = 0.1
        #self.img_rows, self.img_cols = 32, 32

    def run_model(self,model):
        optimizer = optimizers.SGD(lr=0.1, decay=1e-4, momentum=0.9, nesterov=True)
        #optimizer = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        model.compile(loss=losses.binary_crossentropy, optimizer=optimizer, metrics=["accuracy"])
        self.start_time = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
        self.history = model.fit(
            self.trainX,
            self.trainY,
            batch_size=self.batch_size,
            epochs=self.nb_epoch,
            validation_data=(self.testX,self.testY),
            shuffle= True,
            callbacks=[self.lr_reducer,self.tb]
            )
        self.end_time = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
        self.model = model

    
    def run_model_augmentation(self,model):
        optimizer = optimizers.SGD(lr=0.1, decay=1e-4, momentum=0.9, nesterov=True)
        model.compile(optimizer=optimizer, loss=losses.binary_crossentropy, metrics=['acc'])
        self.start_time = datetime.now().strftime("%Y/%m/%d %H:%M:%S")

        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False
        )  # randomly flip image

        datagen.fit(self.trainX)
        model.fit_generator(datagen.flow(self.trainX, self.trainY, batch_size=self.batch_size),
            steps_per_epoch=self.trainX.shape[0] ,
            validation_data=(self.testX,self.testY),
            epochs=self.nb_epoch, verbose=1, max_q_size=100,
            callbacks=[self.lr_reducer,self.tb]
        )
        self.end_time = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
        self.model = model



    def evalute_model(self):
        self.score=self.model.evaluate(self.testX,self.testY,verbose=0,batch_size=self.batch_size)
        print('Test loss:',self.score[0])
        print('Test accuracy:',self.score[1])
        self.accuracy = self.score[1]
        self.loss = self.score[0]

    def get_name(self,global_name):
        method = self.option["block"]
        concanate = self.option["concatenate"]
        double_input = str(self.option["double_input"])
        dropout = str(self.option["dropout"])
        wide=str(self.option["wide"])
        filters = str(self.option["filters"])

        return self.dataset_name+'_'+method+'_'+concanate+'_'+double_input+'_'+dropout+'_'+wide+'_'+filters

    def run(self,model,global_name,argment=False):
        self.lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)
        self.tb = TensorBoard(
            log_dir='./result/tensofboard/'+self.get_name(global_name), 
            histogram_freq=0, 
            batch_size=32, 
            write_graph=True, 
            write_grads=False, 
            write_images=False, 
            embeddings_freq=0, 
            embeddings_layer_names=None, 
            embeddings_metadata=None)
        if argment == False:
            self.run_model(model)
        else:
            self.run_model_augmentation(model)
        self.evalute_model()
        self.model_save(global_name)
        json_writer.json_write(self,'result/'+global_name+'.json')
        K.clear_session()
    
    def model_save(self,global_name):
        print('save models')
        directory_name = 'result/'+global_name+'_models/'
        
        #method = self.option["block"]
        #concanate = self.option["concatenate"]
        #double_input = str(self.option["double_input"])
        #dropout = str(self.option["dropout"])

        #filename =  directory_name+self.dataset_name+'_'+method+'_'+concanate+'_'+double_input+'_'+dropout+'.h5'
        filename = directory_name+self.get_name(global_name)+'.h5'

        if os.path.isdir(directory_name) == False:
            os.makedirs(directory_name)

        self.model.save(filename)

        print('saved model!!('+filename+')')
    

    def __init__(self,option):
        self.option = option
    
    def __del__(self):
        del self.model



def make(option,dataset,batch_size,epochs,split):
    tester = ResNetTester(option = option)
    tester.setDataset(dataset)
    tester.batch_size = batch_size
    tester.nb_epoch = epochs
    tester.validation_split = split
    model = ModelBuilder.ResnetBuilder.build_manual(input_shape=dataset.get_shape(),num_outputs=dataset.get_categorical(), option = option)
    return model,tester

def run(model,tester,global_name,argment=False):
    tester.run(model,global_name,argment=argment)
        