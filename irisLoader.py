from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


class IrisLoader:

    def __init__(self):
        self.data=load_iris()
        self.x=self.data.data
        self.y_=self.data.target.reshape(-1,1)
        self.y=None

    def __str__(self):
        
        return'Example data: \n'+"{} \n".format(self.data.data[:5])+'Example labels: \n'+"{} \n".format(self.data.target[:5])
         
    def encode(self):    
        encoder = OneHotEncoder(sparse=False)
        self.y = encoder.fit_transform(self.y_)
        
    def dataSplit(self,test_size):
        
        return train_test_split(self.x, self.y, test_size=test_size)
    
