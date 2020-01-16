
from modelDataLoader import ModelDataLoader
from rbfModel import RbfModel
import random

class RbfManager:
    def __init__(self):
        print("Welcome to RBF Manager \n")
        print("Loading data\n")
        self.rbfLoader=ModelDataLoader("irisTrainData.csv")
        print("Preprocess data\n")
        target="loss"
        features=["fc1_neurons","fc2_neurons","fc1_activ","fc2_activ","learning_rate"]
        maper={"elu":1,"relu":2,"selu":3,"tanh":4,"sigmoid":5,"exponential":6,"linear":7}
        feature_maper=[("fc1_activ",maper),("fc2_activ",maper)]
        self.rbfLoader.preproces(features,target,feature_maper,estandarize=True) 
        print("Data loaded \n")
        print(self.rbfLoader)

        print("Creating BRF model controller\n")
        self.model=RbfModel(self.rbfLoader)

    
    
    def run(self):
        print("Starting... \n")
      
        print("Compiling model... \n")
        self.model.compileModel()
        print("Model compiled!! \n")
        print("Fiting...\n")
        self.model.fit(0.2)
        

    def save(self):
        self.model.save()