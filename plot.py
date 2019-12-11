# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 14:28:07 2019

@author: xuan
"""
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import random
import matplotlib.pyplot as plt
from dataProcessing import Data

class plot:
    
    def __init__(self, x, y):
        
        self.x = x
        self.y = y
        
    def projLDA(self, X, y, n_components=2):
        """
        function for reducing the dimension for scatter plot. Does not work well
        :param X: the data to trnasform
        :param y: the target
        :param n_components: the number of component to save
        :return: the fitted data in n_components dimension
        """
        lda = LinearDiscriminantAnalysis(n_components=n_components)
        lda.fit(X, y)
        return lda.transform(X)
    

    def randomcolor(self):
        colorArr = ['1','2','3','4','5','6','7','8','9','A','B','C','D','E','F']
        color = ""
        for i in range(6):
            color += colorArr[random.randint(0,14)]
        return "#"+color
        
    def scatterPlot(self):
        """
        function for scatter plot.
        :param X: data to plot, the scaled data from scale() function
        :param y: the target
        :param colorMap: color
        :return: scatter plot to show the class
        """
        X = self.projLDA(self.x, self.y)
        plt.figure()
        plt.figure(figsize=(18,12))
        plt.subplot(1, 1, 1)
        colorlist = set()
        while len(colorlist) <= 99:
            colorlist.add(self.randomcolor())
        colorlist = list(colorlist)
        for i in range(len(self.y)):
            #if self.y[i]>=10 and self.y[i]<17:
            plt.scatter(X[i, 0], X[i, 1], c=colorlist[self.y[i]], s=300, alpha=0.9)
        #plt.scatter(X[:, 0], X[:, 1], c=self.y, s=100, alpha=0.9)
        plt.tick_params(labelsize=30)
        plt.xlabel('Principal1',fontsize=40)
        plt.ylabel('Principal2',fontsize=40)
        plt.savefig('scatter.pdf')
        plt.show()
        
D = Data()
x_train, y_train, _1, _2 = D.getData()
P = plot(x = x_train, y = y_train)
P.scatterPlot()
