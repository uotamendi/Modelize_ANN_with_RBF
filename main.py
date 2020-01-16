from modelDataManager import ModelDataManager
from irisManager import IrisManager

from rbfManager import RbfManager
get_data=False
if get_data:
    ## The filename of the CSV where all the data created by the target ANN will be saved
    filename="irisTrainData.csv"

    ## Initialize model's data manager
    dataManager=ModelDataManager(filename)

    ## Initialize Iris model's manager
    irisManager=IrisManager()

    ## Load training data for 200 iterations
    try: 
        for i in range(200):
            print("----> Iteration Nº{}".format(i))
            hp,loss,acc=irisManager.run()
            dataManager.addRow(hp,loss,acc)
    except :
        dataManager.save()
    dataManager.save()

else:
    rbfManager=RbfManager()
    rbfManager.run()
    rbfManager.save()