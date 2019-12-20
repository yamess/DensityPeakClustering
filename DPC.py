#####################################################
# Author: Boukari Yameogo                       	#
# Date: 2019-12-05                                  #
# Density Peak Clustering						    #
# Team Project				                        #
#####################################################

# Librairies de python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy


def timing(func):
    """
    Cette fonction est un decorateur qui met de calculer le temps d'execution d'une fonction
    :param func: la fonction a évaluer
    :return: exécute la fonction et le temps d'exécution
    """
    import time

    def wrapper(*args, **kwargs):
        print("Execution start")
        start = time.perf_counter()
        res = func(*args, **kwargs)
        end_time = time.perf_counter()
        print(f"Execution time: {round(end_time-start, 2)} second(s)")
        print("Finished!")
        return res
    return wrapper


def sort_n(arr, n_to_sort=1, how='max'):
    """
    Cette fonction permet de retourner le n plus grande ou plus petite
    valeurs d'une liste avec leur indice
    :param arr: La liste à trier
    :param n_to_sort: le nombre d'élement à retourner
    :param how: Descending pour decroissant et Ascending pour Croissant
    :return: la liste de n valeurs avec leurs indices
    """
    from copy import deepcopy
    import numpy as np
    data_val = deepcopy(arr)
    n = len(data_val)
    data_id = list(np.arange(n))
    data = np.transpose([data_val, data_id])
    result = data[:n_to_sort]

    if str(how).upper() == 'MIN':
        for i in range(n_to_sort):
            tmp_val = result[i,0]
            tmp_id = result[i,1]
            swap = False
            for j in range(n_to_sort,n):
                if data[j, 0] < tmp_val:
                    data[j, 0], tmp_val = tmp_val, data[j, 0]  # echanger les valeurs
                    data[j, 1], tmp_id = tmp_id, data[j, 1]  # echanger les id
                    swap = True
                if swap:
                    result[i, 0] = tmp_val
                    result[i, 1] = int(tmp_id)
                    swap = False
        return result
    elif str(how).upper() == 'MAX':
        for i in range(n_to_sort):
            tmp_val = result[i, 0]
            tmp_id = result[i, 1]
            swap = False
            for j in range(n_to_sort,n):
                if data[j, 0] > tmp_val:
                    data[j, 0], tmp_val = tmp_val, data[j, 0]  # echanger les valeurs
                    data[j, 1], tmp_id = tmp_id, data[j, 1]  # echanger les id
                    swap = True
                if swap:
                    result[i, 0] = tmp_val
                    result[i, 1] = int(tmp_id)
                    swap = False
        return result
    else:
        print("how prend uniquement les valeurs 'min' ou 'max'")


def trie_table(table, by=0, descending=True):
    """
    Cette fonction permet de trie toute la table avec plusieurs colonne et definissant la colonne de trie
    :param table: La table à triée
    :param by: Integer - l'indice de la colonne selon laquelle la table doit etre triée
    :param descending: Booléen. Par defaut c'est True pour dire dans l'ordre décroissant
    :return: La table trié selon la colonne by
    """
    n = len(table)
    if descending:
        for i in range(n):
            for j in range(n):
                if table[j][by] < table[i][by]:
                    table[j], table[i] = table[i], table[j]
    elif not descending:
        for i in range(n):
            for j in range(n):
                if table[j][by] > table[i][by]:
                    table[j], table[i] = table[i], table[j]
    else:
        print("Descending is boolean and take True or False")
    return table


class DPC:
    """
    Density Peak Clustering Alghorithm Main Class
    """
    def __init__(self, d_min=0.5, nb_cluster=4, type_distance=2):
        """
        Initialisation des paramètres à la construction de la classe
        :param df: un dataframe de type pandas
        :param d_min: la distanceance minimale (hyperparamètre)
        """
        self._type_distance = type_distance
        self.__cluster_center = 0
        self._d_min = d_min
        self.__nbr_observation = 0
        self.__nbr_feature = 0
        self.__delta = 0
        self._nb_cluster = nb_cluster
        self.eps = 1e-20
        self.__rho = 0
        self.__densite_sup = 0
        self.__clusters = dict()

    def __distance(self, point_a, point_b, dim):
        """
        Fonction de calcul de la distanceance entre 2 points
        :param point_a: le point a (pourrait correspondre à une observation dans un dataframe)
        :param point_b: le point b (pourrait correspondre à une observation dans un dataframe)
        :param dim: Le nombre de coordonnée des points
        :return: Retourne la distanceance entre les 2 points a et b
        """
        dimension = dim
        total = 0
        for i in range(dimension):
            total += pow(abs(point_a[i] - point_b[i]), self._type_distance)
        return pow(total, 1/self._type_distance)

    def rho(self, df, nbr_features, nbr_observation):
        """
        Fonction de calcul de la densité locale de chaque point.
        En d'autre terme c'est le nombre de point situé dans un rayon dc de chaque point
        Ici le point pour lequel la densité est calculée n'est pas inclu
        :return: Retourne une liste contenant la densité local de chacque point et les pointscorrespondant
        """
        dim = nbr_features
        n = nbr_observation
        resultat = list()  # Initialisation de la liste en sortie à vide
        for i in range(n):
            a = df.iloc[i, :]  # obtention des coordonnées du point a
            voisins = list()
            densite = 0
            for j in range(i+1, n):
                b = df.iloc[j, :]  # obtention des coordonnées du point b
                if self.__distance(a, b, self.__nbr_feature) < self._d_min:
                    densite += 1
                    voisins.append(j)  # Ajouter j comme voisin de i
            resultat.append([i, densite, voisins])  # Ajouter le triplet i, la densité locale de i et la liste de
            # ses voisins
        res = trie_table(resultat,by=1,descending=True)  # Nous trions la table
        indice_densite_sup = []
        for k in range(1, len(res)):
            indice_densite_sup.append([res[k-1][0], list(np.transpose(res)[0][:k-1])])  # les points ayant
            # une densité supérieurs
        indice_densite_sup.append([res[-1][0],list(np.transpose(res)[0][:-1])])
        return res, indice_densite_sup

    def delta(self, df, rho, point_de_densite_sup):
        """
        Compute the minimum distance between a given point and all other point with higher localm density
        :param df: Datrame
        :param dim: the dimension or number of features
        :param rho: rho
        :param point_de_densite_sup: list of the point with higher density for each point
        :return: Lisst of the min distance
        """
        dens = rho
        grande_densite = point_de_densite_sup
        result = []
        max_dens = dens[0][1]  # le max est le premier element car la liste est deja triée
        for k,i in enumerate(dens):
            if i[1] == max_dens:
                point_a = df.iloc[i[0], :]
                tmp_dist = []
                for j in dens:
                    point_b = df.iloc[j[0],:]
                    tmp_dist.append(self.__distance(point_a, point_b, self.__nbr_feature))
                tmp_dist_sorted = sort_n(tmp_dist,1,'max')
                delta = tmp_dist_sorted[0][0]
                point = dens[int(tmp_dist_sorted[0][1])][0]
                result.append([i[0],i[1],delta,i[1]*delta,point,grande_densite[k][1]])
            else:
                point_a=df.iloc[i[0],:]
                tmp_dist = []
                if len(grande_densite[k][1])>0:
                    for j in grande_densite[k][1]:
                        point_b = df.iloc[j,:]
                        tmp_dist.append(self.__distance(point_a,point_b,self.__nbr_feature))
                    tmp_dist_sorted = sort_n(tmp_dist,1,'min')
                    delta = tmp_dist_sorted[0][0]
                    point = dens[int(tmp_dist_sorted[0][1])][0]
                    result.append([i[0],i[1],np.round(delta,2),np.round(i[1]*delta,2),point,grande_densite[k][1]])
        return result

    def centres(self, delta):
        """
        Finds the center of the cluster base on the cluster number entered
        :param delta: Delta
        :return: List of cluster centers
        """
        data = delta
        spy = True
        n = len(data) - 1
        while spy:
            spy = False
            for i in range(n):
                if data[i][3] < data[i + 1][3]:
                    tmp = data[i]
                    data[i] = data[i + 1]
                    data[i + 1] = tmp
                    spy = True
        return data[:self._nb_cluster]

    def assignation(self, delta, centres):
        """
        Method to assign the non peak points to the clusters
        :param delta: Delta
        :param centres: Center of clusters
        :return: Cluster membership
        """
        delta = delta
        cluster_centres = [i[0] for i in centres]
        clusters_memders = dict()
        # list_point_non_peak = [i[0] for i in delta if i[0] not in cluster_centres]
        list_total_point = [i[0] for i in delta]
        for i, j in enumerate(cluster_centres):
            clusters_memders[f"C_{i}"] = [j]  # Création des points de départ des cluster

        for k, v in enumerate(list_total_point):
            if v not in cluster_centres:
                point_prec = delta[k][4]
                for ind, val_list in clusters_memders.items():
                    if point_prec in val_list:
                        val_list.append(v)
                        clusters_memders[ind] = val_list
        return clusters_memders

    def decision_graph(self):
        """
        PLot the decision graph to see the optimal number of clusters to choose
        :return: Plot
        """
        data = np.transpose(self.__delta)[[0,3]]
        plt.scatter(data[0], data[1])
        plt.xlabel("Data Point Index")
        plt.ylabel("Delta* Rho")
        plt.title("Decision Graph")
        plt.show()

    def cluster_plot(self,df):
        """
        Plot the clusters and their members
        :param df: dataframe
        :return: Plot
        """
        col = ('grey','blue', 'red', 'orange', 'brown')
        i=0
        for k,val in self.__clusters.items():
            plt.scatter(df.iloc[val,0],df.iloc[val,1],c=col[i],label=k)
            i += 1
        plt.title("Clusters")
        plt.legend()
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.show()

    @timing
    def fit(self, X):
        """
        Function to fit the data X
        :param X: Dataframe to fit
        :return: clusters with members
        """
        df = deepcopy(X)
        self.__nbr_feature = df.shape[1]
        self.__nbr_observation = df.shape[0]
        self.__rho, self.__densite_sup = self.rho(df,self.__nbr_feature,self.__nbr_observation)
        self.__delta = self.delta(df,self.__rho,self.__densite_sup)
        self.__cluster_center = self.centres(self.__delta)
        self.__clusters = self.assignation(self.__delta,self.__cluster_center)
        self.cluster_plot(df)
        return self.__clusters


# TEST DU CODE
# importation du dataset (path à changer selon la source de votre donnée)
dt = pd.read_csv("data/data.csv", sep=";", header=None, usecols=[0, 1])
t = DPC(0.5,5)
d = t.fit(dt)
