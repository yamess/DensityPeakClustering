#####################################################
# Author: Boukari Yameogo & Group                  	#
# Date: 2019-11-05                                  #
# Density Peak Clustering						    #
# Team Project				                        #
#####################################################

# Librairies de python
import numpy as np
import pandas as pd
from copy import deepcopy

# Nos propre librairies
from mypackages.Heapsort import Heap
from mypackages.decorators import timing
from mypackages.processing import sort_n
from mypackages.parallele import parallele


class DPC:
    """
    Clustering algorithm to optimize the
    """
    def __init__(self, df, d_min=0.5, nb_cluster=4, type_distance=2):
        """
        Initialisation des paramètres à la construction de la classe
        :param df: un dataframe de type pandas
        :param d_min: la distance minimale (hyperparamètre)
        """
        self._df = df.copy()
        self._type_distance = type_distance
        self._cluster_center = []
        self._d_min = d_min
        self._nbr_points = self._df.shape[0]
        self._nbr_dim = self._df.shape[1]
        self._densite = []
        self._nb_cluster = nb_cluster
        self.eps = 1e-20  # valeur d'epsilon pour comparer à 0

    @property
    def cluster_center(self):
        return self.cluster_center

    @cluster_center.setter
    def cluster_center(self, value):
        self.cluster_center = value

    def dist(self, point_a, point_b):
        """
        Fonction de calcul de la distance entre 2 points
        :param point_a: le point b (pourrai correspondre à une observation dans un dataframe)
        :param point_b: le point b (pourrai correspondre à une observation dans un dataframe)
        :return: Retourne la distance euclidienne entre les 2 points point_a et point_b
        """
        dimension = self._nbr_dim
        total = 0
        for i in range(dimension):
            total += pow(abs(point_a[i] - point_b[i]), self._type_distance)
        return pow(total, 1/self._type_distance)

    def func_densite(self):
        """
        Fonction de calcul du nombre point situé dans un rayon de d_min d chaque point
        Ici le point pour lequel le calcul est fait n'est pas inclu
        :return: Retourne un liste contenant la densité pour chaque point
        """
        n = self._nbr_points
        dens = np.zeros(n, dtype=int)  # on defini un liste vide
        result = list()  # Initialisation de la liste en sortie à vide
        for i in range(n):
            a = self._df.iloc[i, :]  # obtention des coordinnées du point a
            for j in range(i+1, n):
                b = self.df.iloc[j, :]  # obtention des coordinnées du point b
                if self.dist(point_a=a, point_b=b) < self._d_min:
                    dens[i] += 1
            result.append([i, dens[i]])
        return result

    def dist_min_grde_densite(self, dens):
        """
        Cette fonction permet de trouver la distance minimale
        des point de grandes densités
        :param dens: la table contenant la liste des points et leur densité
        :return: une liste contenant le distance minimale
        """
        result = []
        max_dens = sort_n(np.transpose(dens)[1], 1, 'max')
        for i, couplet in enumerate(dens):
            densite = couplet[1]
            pt_a = self._df.iloc[couplet[0], :]
            if densite == max_dens:
                tmp_dist = []
                for j in range(len(dens)):
                    pt_b = self._df.iloc[j, :]
                    tmp_dist.append(self.dist(point_a=pt_a, point_b=pt_b))
                rho = sort_n(tmp_dist, 1, 'max')
                result.append([couplet[0], couplet[1], round(rho, 2)])
            else:
                tmp_dist = []
                for j in range(len(dens)):
                    if couplet[1] < dens[j][1]:
                        pt_b = self.df.iloc[j, :]
                        tmp_dist.append(self.dist(point_a=pt_a, point_b=pt_b))
                rho = sort_n(tmp_dist, 1, 'min')
                result.append([couplet[0], couplet[1], round(rho, 2)])
        return result

    def clusters_centers(self, dens):
        """
        :param dens:
        :return:
        """
        d = dens.copy()
        spy = True
        n = len(d) - 1
        while spy:
            spy = False
            for i in range(n):
                if d[i][3] > d[i + 1][3]:
                    tmp = d[i]
                    d[i] = d[i + 1]
                    d[i + 1] = tmp
                    spy = True
        return d[-self._nb_cluster:]

    def assignation(self, dens, cluster_center):
        tmp = self._df
        for i in range(self._nbr_points):
            if i not in np.transpose(cluster_center)[0]:
                point_a = self.df.iloc[i, :]
                tmp = []
                pos = []
                for j in np.transpose(cluster_center)[0]:
                    centre = self.df.iloc[j, :]
                    tmp.append([self.dist(point_a, centre), j])
                min = sort_n(np.transpose(tmp)[1], 1, how='min')


@timing
def main():
    df = pd.read_csv("data/data.csv", sep=";", header=None, usecols=[0, 1])
    pc = DPC(df, 0.5, 5)
    dens = pc.func_densite()
    result = pc.dist_min_grde_densite(dens)
    clusters = cluster_centers_weigth(result)
    centers = pc.clusters_centers(clusters)
    print(f"Dimension de df: {df.shape}\n")
    print(f"Nombre de valeur de densité: {len(dens)}\n")
    print(f"Nombre de valeur des triplets: {len(result)}\n")
    print(f"{dens}\n")
    print(f"{result}\n")
    print(f"{clusters}")
    print(f"{centers}")


if __name__ == "__main__":
    main()