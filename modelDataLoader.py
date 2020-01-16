import pandas as pd
from sklearn.model_selection import train_test_split



class ModelDataLoader:

    def __init__(self,filename):
        self.filename=filename
        self.data=pd.read_csv(filename)
        self.x = None
        self.y = None

    def preproces(self,features,target,discretization_factor=[],estandarize=False):
        print("Starting data preprocesing...\n")
        print("Discretizating columns...\n")
        self.discretize(discretization_factor)
        print("Separating features and target..\n")
        self.separate_data(features,target)
        if estandarize:
            print("Standardizing..\n")
            self.estandardize()

    def separate_data(self,features,target):
        columns=self.data.columns
        if not(target in columns):
            print("Target column does not exist")
            return -1
        self.y=self.data[target].values
        
        for column in columns:
            if not(column in features):
                self.data=self.data.drop([column],axis=1) 
        
        
        self.x=self.data.values
    

    def discretize(self,features):
        for (name,discretization) in features:
            self.data[name]=self.data[name].map(discretization)
        

    def estandardize(self):
        self.x=(self.x-self.x.mean())/self.x.std()
        

    def __str__(self):
        
        return'Example data: \n'+"{} \n".format(self.x[0])
         
    
        
    def dataSplit(self,test_size):
        
        return train_test_split(self.x, self.y, test_size=test_size)
    
