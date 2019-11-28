# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 12:04:17 2019

@author: xuan
"""
from data import Data

class ensemblestudy(Data):
    def __init__(self, weight):
        super().__init__()
        self.weight = weight
        