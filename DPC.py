#####################################################
# Author: Boukari Yameogo & Group                  	#
# Date: 2019-11-05                                  #
# Density Peak Clustering						    #
# Team Project				                        #
#####################################################

import numpy as np
import pandas as pd
from copy import deepcopy
from mypackages.Heapsort import Heap


def timing(func):
    """
    Cette fonction est un decorateur qui met de calculer le temps d'execution d'une fonction
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


def parallele(func, arg, max_worker):
    """
    Cett fonction permet de paralleliser les taches de la fonction f
    sur les arguments arg données
    :param func: Fonction de tache
    :param arg: les arguments de la fonction f à évaluer
    :param max_worker: Nombre total de thread à lançer
    :return: Liste des resultats de l'exécution de la fonction f sur pour chaque argument de arg
    """
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_worker=max_worker) as executor:
        result = executor.map(func, arg, timeout=None)
    return result


def cluster_centers_weigth(dens):
    """

    :param dens:
    :return:
    """
    result = []
    for i, couplet in enumerate(dens):
        center = dens[i][1]*dens[i][2]
        result.append([couplet[0], couplet[1], couplet[2], round(center, 2)])
    return result


def sort_n(arr, n_to_sort=1, how='max'):
    """
    Cette fonction permet à de retourner les n valeur max/min d'une list
    :param arr: la liste à trier
    :param n_to_sort: n nombre à retourner
    :param how: Préciser si minimum ou maximum
    :return: n valeur max/min
    """
    arr_tmp = deepcopy(arr)
    n = len(arr)
    if n_to_sort == 1:
        res = deepcopy(arr_tmp[0])
        if how == 'max':
            for j in range(n_to_sort,n):
                if arr_tmp[j] > res:
                    res, arr_tmp[j] = arr_tmp[j], res
        elif how == 'min':
            for j in range(n_to_sort,n):
                if arr_tmp[j] < res:
                    res, arr_tmp[j] = arr_tmp[j], res
        return res
    else:
        res = deepcopy(arr_tmp[:n_to_sort])
        swap = False
        if how == 'max':
            for i in range(n_to_sort):
                tmp = deepcopy(res[i])
                for j in range(n_to_sort,n):
                    if arr_tmp[j] > tmp:
                        tmp, arr_tmp[j] = arr_tmp[j], tmp
                        swap = True
                if swap:
                    res[i] = tmp
                    swap = False
            return res
        elif how == 'min':
            for i in range(n_to_sort):
                tmp = deepcopy(arr_tmp[i])
                for j in range(n_to_sort,n):
                    if arr_tmp[j] < tmp:
                        tmp, arr_tmp[j] = arr_tmp[j], tmp
                        swap = True
                if swap:
                    res[i] = tmp
                    # resultat.append(tmp)
                    swap = False
            return res


class DPC:
    """
    Clustering algorithm to optimize the
    """
    def __init__(self, df, d_min=0.5, nb_cluster=4, ):
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
        self.nb_cluster = nb_cluster
        self.eps = 1e-20  # valeur d'epsilon pour comparer à 0
        # self.get_params = dict()
        # self.heap = Heap()

    # @property
    # def get_params(self):
    #     return self.get_params
    #
    # @get_params.setter
    # def get_params(self, value):
    #     self.get_params = value

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
        n = self.nbr_points
        dens = np.zeros(n, dtype=int)  # on defini un liste vide
        result = list()  # Initialisation de la liste en sortie à vide
        for i in range(n):
            a = self.df.iloc[i, :]  # obtention des coordinnées du point a
            for j in range(i+1, n):
                b = self.df.iloc[j, :]  # obtention des coordinnées du point b
                if self.dist(point_a=a, point_b=b) < self.d_min:
                    dens[i] += 1
            result.append([i, dens[i]])
        return result

    # def comptage(self, array, val):
    #     """
    #     Fonction de comptage d'un nombre val dans la list array
    #     :param array: array dans lequel la recherche doit s'effectuer
    #     :param val: la valeur dont on veut compter l'occurence
    #     :return: Retourne le nombre total d'occurence de val dans array
    #     """
    #     n = len(array)
    #     nbr = 0
    #     for i in range(n):
    #         if (array[i] - val) > self.eps:
    #             nbr += 1
    #     return nbr

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
            pt_a = self.df.iloc[couplet[0], :]
            if densite == max_dens:
                tmp_dist = []
                for j in range(len(dens)):
                    pt_b = self.df.iloc[j, :]
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
        return d[-self.nb_cluster:]

    def assignation(self, dens, cluster_center):
        for i in range(self.nbr_points):
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

a = [[94, 5, 6.96, 34.8], [19, 7, 10.32, 72.24], [15, 7, 12.91, 90.37], [0, 7, 13.2, 92.4], [6, 8, 19.94, 159.52]]
np.transpose(a)[0]
import numpy as np
b = []
b.append([12,1])
b.append([3,6])
b.append([5,9])
np.transpose(b)[0]