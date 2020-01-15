class Hyperparameter:

    def __init__(self):
        self.hyper={}

    def add(self,name,value):
        self.hyper[name]=value

    def get(self,name):
        if name in self.hyper:
            return self.hyper[name]

        return None

    def __str__(self):
        print("Hyperparameters \n")
        for i in self.hyper:
            print("- {} -> {} \n".format(i,self.get(i)))
        