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
    Cett fonction permet de paralleliser les tache de la fonction f
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


class Process:
    """
    Clustering algorithm to optimize the
    """
    def __init__(self, df, d_min, nb_cluster=4):
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
        result = list()
        for i in range(n):
            a = self.df.iloc[i, :]
            for j in range(i+1, n):
                b = self.df.iloc[j, :]
                if self.dist(point_a=a, point_b=b) < dmin:
                    dens[i] += 1
            result.append([i, dens[i]])
        return result

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

    def max_val(self, array):
        """
        Cette fonction return la valeur maximal dans une liste array
        en appliquant un seul iteration du tri a bulle
        :param array: la liste à evaluer
        :return: la valeur maximale de la liste
        """
        n = len(array)
        max_val = array[0]
        for i in range(1, n):
            if array[i] > max_val:
                max_val = array[i]
        return max_val

    def min_val(self, array):
        """
        Fonction pour trouver la plus petite valeur dans une liste
        :param array: la liste à évaluer
        :return: la plus petite valer de la liste
        """
        n = len(array)
        min_val = array[0]
        for i in range(1, n):
            if array[i] < min_val:
                min_val = array[i]
        return min_val

    def dist_min_grde_densite(self, dens):
        """
        Cette fonction permet de trouver la distance minimale
        des point de grandes densités
        :param dens: la table contenant la liste des points et leur densité
        :return: une liste contenant le distance minimale
        """
        result = []
        max_dens = self.max_val(np.transpose(dens)[1])
        for i, couplet in enumerate(dens):
            densite = couplet[1]
            pt_a = self.df.iloc[couplet[0], :]
            if densite == max_dens:
                tmp_dist = []
                for j in range(len(dens)):
                    pt_b = self.df.iloc[j, :]
                    tmp_dist.append(self.dist(point_a=pt_a, point_b=pt_b))
                rho = self.max_val(tmp_dist)
                result.append([couplet[0], couplet[1], round(rho, 2)])
            else:
                tmp_dist = []
                for j in range(len(dens)):
                    if couplet[1] < dens[j][1]:
                        pt_b = self.df.iloc[j, :]
                        tmp_dist.append(self.dist(point_a=pt_a, point_b=pt_b))
                rho = self.min_val(tmp_dist)
                result.append([couplet[0], couplet[1], round(rho, 2)])
        return result

    def cluster_centers_weigth(self, dens):
        result = []
        for i, couplet in enumerate(dens):
            center = dens[i][1]*dens[i][2]
            result.append([couplet[0], couplet[1], couplet[2], round(center, 2)])
        return result

    def clusters_centers(self, dens):
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


@timing
def main():
    df = pd.read_csv("data.csv", sep=";", header=None, usecols=[0, 1])
    p = Process(df, 0.5, 5)
    dens = p.func_densite()
    result = p.dist_min_grde_densite(dens)
    clusters = p.cluster_centers_weigth(result)
    centers = p.clusters_centers(clusters)
    print(f"Dimension de df: {df.shape}\n")
    print(f"Nombre de valeur de densité: {len(dens)}\n")
    print(f"Nombre de valeur des triplets: {len(result)}\n")
    print(f"{dens}\n")
    print(f"{result}\n")
    print(f"{clusters}")
    print(f"{centers}")


if __name__ == "__main__":
    main()
