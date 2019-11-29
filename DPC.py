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
import pyximport

# Nos propre librairies
# from mypackages.heapsort import Heap
from mypackages.decorators import timing
from mypackages.processing import sort_n, cluster_centers_weigth
from mypackages.parallele import parallele


class DPC:
    """
    Clustering algorithm to optimize the
    """
    def __init__(self, df, d_min=0.5, nb_cluster=4, type_distanceance=2):
        """
        Initialisation des paramètres à la construction de la classe
        :param df: un dataframe de type pandas
        :param d_min: la distanceance minimale (hyperparamètre)
        """
        self._df = df.copy()
        self._type_distanceance = type_distanceance
        self._cluster_center = []
        self._d_min = d_min
        self._nbr_points = self._df.shape[0]
        self._nbr_dim = self._df.shape[1]
        self._densite = []
        self._nb_cluster = nb_cluster
        self.eps = 1e-20  # valeur d'epsilon pour comparer à 0
        self.point = None
        # self.rho = 0

    def distance(self, point_a, point_b):
        """
        Fonction de calcul de la distanceance entre 2 points
        :param point_a: le point b (pourrai correspondre à une observation dans un dataframe)
        :param point_b: le point b (pourrai correspondre à une observation dans un dataframe)
        :return: Retourne la distanceance euclidienne entre les 2 points point_a et point_b
        """
        dimension = self._nbr_dim
        total = 0
        for i in range(dimension):
            total += pow(abs(point_a[i] - point_b[i]), self._type_distanceance)
        return pow(total, 1/self._type_distanceance)

    def rho(self):
        """
        Fonction de calcul de la densité locale de chaque point.
        En d'autre terme c'est le nombre de point situé dans un rayon dc de chaque point
        Ici le point pour lequel la densité est calculée n'est pas inclu
        :return: Retourne une liste contenant la densité local de chacque point et les pointscorrespondant
        """
        n = self._nbr_points
        # densite = 0  # on defini une liste vide
        resultat = list()  # Initialisation de la liste en sortie à vide
        for i in range(n):
            a = self._df.iloc[i, :]  # obtention des coordinnées du point a
            ensemble_point = list()
            densite = 0
            for j in range(i+1, n):
                b = self._df.iloc[j, :]  # obtention des coordinnées du point b
                if self.distance(point_a=a, point_b=b) < self._d_min:
                    densite += 1
                    ensemble_point.append(j)
            resultat.append([i, densite, ensemble_point])
        for i in range(n):
            for j in range(n):
                if resultat[i][1] > resultat[j][1]:
                    resultat[i], resultat[j] = resultat[j], resultat[i]
        return resultat

    # @rho.setter
    # def rho(self, value):
    #     self.rho = value
    @property
    def delta(self):
        """
        Cette fonction permet de caclculer delta qui la distance minimale entre un point donné et
        les points qui ont une plus grande densité que ce point
        :param dens: la table contenant la liste des points et leur densité, la sortie de la fonction rho
        :return: une liste contenant le distanceance minimale
        """
        dens = self.rho()
        result = []
        max_dens = dens[0] # le max est le premier element car la liste est deja triée
        for i, n_uplet in enumerate(dens):
            densite = n_uplet[1]
            pt_a = self._df.iloc[n_uplet[0], :]
            if densite == max_dens:
                tmp_distance = []
                for j in range(len(dens)):
                    pt_b = self._df.iloc[j, :]
                    tmp_distance.append(self.distance(point_a=pt_a, point_b=pt_b))
                if len(tmp_distance) > 1:
                    dist_max = sort_n(tmp_distance, 1, 'max')[0][0]
                else:
                    dist_max = tmp_distance
                result.append([n_uplet[0], n_uplet[1], np.round(dist_max,2), n_uplet[2]])
            elif len(dens[i][2]) > 0:
                tmp_distance = []
                for j in dens[i][2]:
                    pt_b = self._df.iloc[j,:]
                    tmp_distance.append(self.distance(point_a=pt_a,point_b=pt_b))
                if len(tmp_distance) > 1:
                    dist_min = sort_n(tmp_distance,1,'max')[0][0]
                else:
                    dist_min = tmp_distance[0]
                result.append([n_uplet[0], n_uplet[1], np.round(dist_min,2), n_uplet[2]])
            else:
                result.append([n_uplet[0], n_uplet[1], 0.0, n_uplet[2]])
        return result

    def centres(self):
        """
        :param dens:
        :return:
        """
        # result = []
        data = self.delta
        data = [[data[i][0],data[i][1], data[i][2], np.round(data[i][1]*data[i][2],2),data[i][3]]
                for i, _ in enumerate(data)]
        # d = self.delta.copy()
        spy = True
        n = len(data) - 1
        # cluster_ind = []
        while spy:
            spy = False
            for i in range(n):
                if data[i][3] < data[i + 1][3]:
                    tmp = data[i]
                    data[i] = data[i + 1]
                    data[i + 1] = tmp
                    spy = True
        return data[:self._nb_cluster]

    @property
    def point_assigner(self):
        return list(np.transpose(self.delta)[0])

    @point_assigner.setter
    def point_assigner(self, value):
        pass

    def assignation(self):
        cluster_centres = [i[0] for i in self.centres()]
        clusters_memders = dict()
        list_total = self.point_assigner
        list_point = [i for i in list_total if i not in cluster_centres]
        for i, j in enumerate(cluster_centres):
            clusters_memders[i] = [j]
        for i in list_point:
            list_sup = list_total[:list_total.index(i)]
            dist = []
            for k in list_sup:
                dist.append(self.distance(self._df.iloc[i, :], self._df.iloc[k, :]))
            # dist_min = sort_n(dist, n_to_sort=1, how='min')
            point_pred = int(sort_n(dist, n_to_sort=1, how='min')[0][1])

            # int(dist_min[0][1])
            # if point_pred in cluster_centres:
            #     clusters_memders[cluster_centres.index(point_pred)].append(i)
            # else:
            for k, v in clusters_memders.items():
                if point_pred in v:
                    clusters_memders[k].append(i)
                    # else:
                    #     clusters_memders[max(clusters_memders.keys())+1].append(i)
                        # break
                        # print(i,k,v)
        return clusters_memders

    # def decision_graph(self):
    #     import matplotlib.pyplot as plt
    #     clusters = self.assignation()
    #
    #     colors = ["green","blue","red","yellow"]
    #     for k, v in clusters.items():
    #         plt.scatter(x=self._df.iloc[v, 0],
    #                     y=self._df.iloc[v, 1],
    #                     color=colors[k],
    #                     alpha=0.6,
    #                     marker="o")
    #     plt.show()
    #     # data = np.transpose(self.delta)[[0,1,2]]
        # plt.scatter(data[1], data[2])
        # plt.xlabel("Densité Local rho")
        # plt.ylabel("Delta")
        # plt.title("Decision Graph")
        # plt.show()

# @timing
# def main():
df = pd.read_csv("data/data.csv", sep=";", header=None, usecols=[0, 1])
pc = DPC(df, 0.5, 5)
# print(pc.point_assigner)
# print(pc.rho())
# print(pc.delta)
# pc.decision_graph()
# d = pc.centres()
test = pc.assignation()
#     clusters = cluster_centers_weigth(result)
#     centers = pc.clusters_centers(clusters)
#     print(f"Dimension de df: {df.shape}\n")
#     print(f"Nombre de valeur de densité: {len(dens)}\n")
#     print(f"Nombre de valeur des triplets: {len(result)}\n")
#     print(f"{dens}\n")
#     print(f"{result}\n")
#     print(f"{clusters}")
#     print(f"{centers}")


# if __name__ == "__main__":
#     main()

# print(test.keys())
print(len(test[0])+len(test[1])+len(test[2])+len(test[3])+len(test[4]))
# import matplotlib.pyplot as plt
# plt.scatter(df.iloc[:,0],df.iloc[:,1])
# plt.show()