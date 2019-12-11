# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 12:04:17 2019

@author: xuan
"""
from algorithmEvaluation import AlgorithmEvaluation
import numpy as np
from plot import plot

"""
There are some little modification in other files.In 
algorithmEvaluation.py, add "return weight, mod" in 
algorithmEvaluation function, and in dataProcessing.py 
transfrom self.X_unknown into matrix.
"""

class ensemblestudy(AlgorithmEvaluation):
    def __init__(self):
        
        super().__init__()
        self.weight,self.models = super().algoEvaluation()
    
    def vote(self, result):
        """
            Base on the weight of each algorithms, find out
        the best one in the results from each algorithms.
        """
        recorddic = {}
        dickey = []
        for i in range(len(result)):
            restr = str(result[i])
            if restr not in dickey:
                recorddic[restr] = self.weight[i]
            else:
                recorddic[restr] += self.weight[i]
            dickey.append(restr)
        maxkey = max(recorddic, key=recorddic.get)
        return maxkey[1:len(maxkey)-1]
    
    def results(self):
        """
        
        """
        reli = []
        X_train = np.vstack((self.X_train, self.X_test))
        y_train = np.hstack((self.y_train, self.y_test))
        X_test = self.X_unknown
        for model in self.models:
           
            model.fit(X_train,y_train)

        for i in range(X_test.shape[0]):
            #print(X_test[i,:])
            
            result = []
            for model in self.models:
                result.append(model.predict(np.reshape(X_test[i,:],(1,-1))))
            #reli.append(self.className[int(self.vote(result))])
            reli.append(int(self.vote(result)))
        return X_test, reli
        #print(reli)
            
en = ensemblestudy()
x, y = en.results()
P = plot(x = x, y = y)
P.scatterPlot()
