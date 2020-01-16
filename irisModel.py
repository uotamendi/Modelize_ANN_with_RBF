
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras import backend as K
class IrisModel:

    def __init__(self,dataLoader):
        self.model=None
        self.dataLoader=dataLoader
        
    def compileModel(self,hp):
        K.clear_session()
        del self.model
        self.model= Sequential()
        
        
        self.model.add(Dense(hp.get("fc1_neurons"), input_shape=(4,), activation=hp.get("fc1_activ"), name='fc1'))
        
        self.model.add(Dense(hp.get("fc2_neurons"), activation=hp.get("fc2_activ"), name='fc2'))
        
        self.model.add(Dense(3, activation='softmax', name='output'))

        # Adam optimizer with learning rate of 0.001
        optimizer = Adam(lr=hp.get("learning_rate"))
        self.model.compile(optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    def fit(self,test_size):
        train_x, test_x, train_y, test_y = self.dataLoader.dataSplit(test_size)
        self.model.fit(train_x, train_y, verbose=0, batch_size=5, epochs=200)

        results = self.model.evaluate(test_x, test_y)

        print('Final test set loss: {:4f}'.format(results[0]))
        print('Final test set accuracy: {:4f}'.format(results[1]))
        return results[0],results[1]