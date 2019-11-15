#####################################################
# Author: Boukari Yameogo                           #
# Date: 2019-11-05                                  #
# This class is to verify if the dataset meets      #
# the clustering conditions                         #
#####################################################

import numpy as np
import pandas as pd


def timing(func):
    """
    Cette fonction est un decorateur qui permet de calculer le temps d'execution d'une fonction
    :param func: la fonction a évaluer
    :return: exécute la fonction et le temps d'exécution
    """
    import time

    def wrapper(*args, **kwargs):
        print("Calculation start")
        start = time.perf_counter()
        res = func(*args, **kwargs)
        end_time = time.perf_counter()
        print(f"Execution time: {round(end_time-start,2)} second(s)")
        print("Finished!")
        return res
    return wrapper


class Process:
    """
    Clustering algorithm to optimize the
    """
    def __init__(self, df, d_min):
        """
        Initialisation des paramètres à la construction de la classe
        :param df: un dataframe de type pandas
        :param d_min: la distance minimale (hyperparamètre)
        """
        self.df = df
        self.d_min = d_min
        self.nbr_points = self.df.shape[0]
        self.nbr_dim = self.df.shape[1]
        self.densite = []
        self.pos = []
        self.eps = 1e-20 # valeur d'epsilon pour comparer à 0

    def dist(self, point_a, point_b):
        """
        Fonction de calcul de la distance euclidienne entre 2 points
        :param point_a: le point b (pourrai correspondre à une observation dans un dataframe)
        :param point_b: le point b (pourrai correspondre à une observation dans un dataframe)
        :return: Retourne la distance euclidienne entre les 2 points point_a et point_b
        """
        dimension = self.nbr_dim
        total = 0
        for i in range(dimension):
            total += (point_a[i] - point_b[i])**2
        return np.sqrt(total)

    def func_densite(self):
        """
        Fonction de calcul du nombre point situé dans un rayon de d_min d chaque point
        Ici le point pour lequel le calcul est fait n'est pas inclu
        :return: Retourne un liste contenant la densité pour chaque point
        """
        dmin = self.d_min
        n = self.nbr_points
        dens = np.zeros(n, dtype=int)  # on defini un liste
        for i in range(n):
            a = self.df.iloc[i, :]
            for j in range(i+1, n):
                b = self.df.iloc[j, :]
                if self.dist(point_a=a, point_b=b) < dmin:
                    dens[i] += 1
                    dens[j] += 1
        return dens

    def comptage(self, array, val):
        """
        Fonction de comptage d'un nombre dans un array
        :param array: array dans lequel la recherche doit s'effectuer
        :param val: la valeur dont on veut compter l'occurence
        :return: Retourne le nombre total d'occurence de val dans array
        """
        n = len(array)
        nbr = 0
        for i in range(n):
            if (array[i] - val) > self.eps:
                nbr += 1
        return nbr

    # def peak(self):
    #     rhos = self.func_densite()
    #     n = len(rhos)
    #     peak = list(np.zeros(n, dtype=int))
    #     unique_val = np.unique(rhos)
    #     for i in unique_val:


@timing
def main():
    df = pd.read_csv("data.csv", sep=";", header=None, usecols=[0, 1])
    p = Process(df, 0.5)
    dens = p.func_densite()
    print(f"Dimension de df: {df.shape}\n")
    print(f"Nombre de valeur de densité: {len(dens)}\n")
    print(dens)


if __name__ == "__main__":
    main()
