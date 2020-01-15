import pandas as pd
import os.path

class ModelDataManager:

    def __init__(self,filename):
        self.filename=filename
        if os.path.isfile(filename):
            print("Loading {} file".format(filename))
            self.dataframe=pd.read_csv(filename)
        else:
            print("Creating file...")
            self.dataframe=pd.DataFrame(columns=["fc1_neurons","fc2_neurons","fc1_activ","fc2_activ","learning_rate","loss"])


    def save(self):
        self.dataframe.to_csv(self.filename)

    def addRow(self,hyperparam,loss):
        print("Adding new Row...\n")
        data=hyperparam.getDict()+{"loss":loss}
        print(data)
        self.dataframe.append(data,ignore_index=True)
        print("New Row added!!\n")
