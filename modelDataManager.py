import pandas as pd
import os.path

class ModelDataManager:

    def __init__(self,filename):
        self.filename=filename
        self.columns_hp=["fc1_neurons","fc2_neurons","fc1_activ","fc2_activ","learning_rate"]
        self.columns_target=["loss","acc"]
        if os.path.isfile(filename):
            print("Loading {} file".format(filename))
            self.dataframe=pd.read_csv(filename)
        else:
            print("Creating file...")
            self.dataframe=pd.DataFrame(columns=self.columns_hp+self.columns_target)
        

    def save(self):
        print("Dataframe saved!!!")
        self.dataframe.to_csv(self.filename,index=False)

    def addRow(self,hyperparam,loss,acc):
        print("Adding new Row...\n")
        data=[]
        for column in self.columns_hp:
            data.append(hyperparam.get(column))
           
        data.append(loss)
        data.append(acc)

        print(data)
        self.dataframe=self.dataframe.append(pd.DataFrame([data],columns=self.columns_hp+self.columns_target),sort=False)
        print("New Row added!!\n")
        print(self.dataframe.head())
