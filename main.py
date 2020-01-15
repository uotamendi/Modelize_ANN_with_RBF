from modelDataManager import ModelDataManager
from irisManager import IrisManager
## The filename of the CSV where all the data created by the target ANN will be saved
filename="irisTrainData.csv"

## Initialize model's data manager
dataManager=ModelDataManager(filename)

## Initialize Iris model's manager
irisManager=IrisManager()

## Load training data for 200 iterations
for i in range(200):
    hp,loss=irisManager.run()
    dataManager.addRow(hp,loss)
    
dataManager.save()