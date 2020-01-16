
from irisLoader import IrisLoader
from hyperparameter import Hyperparameter
from irisModel import IrisModel
import random

class IrisManager:
    def __init__(self):
        print("Welcome to Iris Manager \n")
        print("Loading data\n")
        self.irisLoader=IrisLoader()
        print("Encoding data\n")
        self.irisLoader.encode() 
        print("Data loaded \n")
        print(self.irisLoader)

        print("Creating ANN model controller\n")
        self.model=IrisModel(self.irisLoader)

    def generateHP(self):
        activations=["elu","relu","selu","tanh","sigmoid","exponential","linear"]
        print("Generating Hyperparameters \n")
   
        hp= Hyperparameter()
        hp.add("fc1_neurons",random.randint(5,50))
        hp.add("fc2_neurons",random.randint(5,50))

        hp.add("fc1_activ",activations[random.randint(0,len(activations)-1)])
        hp.add("fc2_activ",activations[random.randint(0,len(activations)-1)])
        
        hp.add("learning_rate",random.uniform(0.0001,0.01))

        print("Finished generating hyperparameters \n")
        print(hp)
        return hp
    
    def run(self):
        print("Starting... \n")
        hp=self.generateHP()
        print("Compiling model... \n")
        self.model.compileModel(hp)
        print("Model compiled!! \n")
        print("Fiting...\n")
        loss,acc = self.model.fit(0.2)
        return hp,loss,acc
