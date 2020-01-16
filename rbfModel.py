
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras import backend as K
from keras.regularizers import l1
from keras.layers import LeakyReLU
from datetime import datetime
from keras.models import load_model
import tensorflow as tf
from tensorflow import keras
import numpy as np
import os.path
class RbfModel:

    def __init__(self,dataLoader):
        self.model=None
        self.dataLoader=dataLoader
        self.input_size=len(self.dataLoader.x[0])
        self.logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        self.tensorboard_callback = keras.callbacks.TensorBoard(log_dir=self.logdir)

        
    def compileModel(self):
        K.clear_session()
        del self.model

        if os.path.isfile("rbfModel.h5"):
            self.loadModel()
        else:
            self.model = Sequential()
            

            self.model.add(Dense(4, input_shape=(self.input_size,), activation=LeakyReLU(alpha=0.1), activity_regularizer=l1(0.001), name='fc1'))
            
            self.model.add(Dense(8, activation=LeakyReLU(), name='fc2'))
            
            self.model.add(Dense(4, activation='relu', activity_regularizer=l1(0.001), name='fc3'))
            
            
            self.model.add(Dense(1))

            # Adam optimizer with learning rate of 0.001
            optimizer = Adam(lr=0.001)
            self.model.compile(optimizer, loss='mean_squared_error', metrics=['mse'])


    def fit(self,test_size):
        train_x, test_x, train_y, test_y = self.dataLoader.dataSplit(test_size)
        
        training_history = self.model.fit(
            train_x, # input
            train_y, # output
            batch_size=16,
            verbose=1,
            epochs=100,
            validation_data=(test_x, test_y),
            callbacks=[self.tensorboard_callback],
        )


        print("Average validation loss: ", np.average(training_history.history['val_loss']))
        print("Average validation mse: ", np.average(training_history.history['val_mse']))
        
        return np.average(training_history.history['val_loss']), np.average(training_history.history['val_mse'])
    
    def loadModel(self):    
        self.model=load_model("rbfModel.h5")

    def save(self):
        self.model.save("rbfModel.h5")