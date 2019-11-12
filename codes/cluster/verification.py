#####################################################
# Author: Boukari Yameogo                           #
# Date: 2019-11-05                                  #
# This class is to verify if the dataset meets      #
# the clustering conditions                         #
#####################################################

# importation of the necessary packages
import pandas as pd
import numpy as np


class Verification:
    """
    Verification of the clustering conditions
    Condition 1: The input matrix should be a squared matrice
    Condition 2: The diagonal elements of the matrix should all be null
    Condition 3: The input matrix should be symetric
    Condition 4: The matrix should meet the trinagular criteria
    """
    def __init__(self, data):
        self.matrix = data # We is considering data to be a pandas dataframe
        self.size = data.shape
        self.square = False  # Condition 1
        self.diag = False  # Condition 2
        self.symetry = False  # Condition 3
        self.triangle_equality = False  # Condition 4
        self.eps = 1e-20 # epsilon pour verifier si un nombre decimal peut etre consideré comme nul

    def squared(self):
        result = False
        if self.size[0] == self.size[1]:
            result = True
        return result

    def diag(self):
        n = self.size[0]
        diag = 0
        for i in range(n):
            diag += self.matrix[i, i]
        if diag == 0:
            return False
        else:
            return True

    def symetry(self):
        n = self.size[0]
        for i in range(n-1):
            for j in range(i+1, n):
                if self.matrix[i, j]-self.matrix[j,i] > self.eps:
                    return False
        return True

    def triangle_equality(self):

