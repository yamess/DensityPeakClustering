import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import sklearn
import matplotlib
import urllib
import csv

import pip
#pip.main(["install", "--upgrade", "pip"])

#pip.main(["install", "--upgrade", "pyprind"])

#df = pandas.read_csv("data.csv", delimiter=";")
# lECTURE DU FICHIER SOUS FORMAT dataframe
d = pd.read_csv("../data/data.csv", sep=';', header=None)

# l a3ieme colonne non utilisée dans l'algorithm
# elle sert juste à valider les resultats


# Analyse exploratoire-------------------------------------------------------
# Affichage des données
print(d)
y=d[:,2]
print(np.unique(y))

#print(df.head(3))

#from matplotlib import pyplot as plt

plt.figure(figsize=(10,8))

for clusterId, color in zip(range(0, 5), ('grey','blue', 'red', 'orange', 'brown')):
    plt.scatter(x=d[:,0][d[:,2]== clusterId],
                y=d[:,1][d[:,2]== clusterId],
                color=color,
                marker='o',
                alpha=0.6
                )
plt.title('Clustered Dataset', size=20)
plt.xlabel('$x_1$', size=25)
plt.xlabel('$x_2$', size=25)

plt.tick_params(axis='both', which='major', labelsize=18)
plt.xlim(xmin=np.amin(d[:,0])-0.5, xmax=np.amax(d[:,0])+0.5)
plt.xlim(xmin=np.amin(d[:,1])-0.5, xmax=np.amax(d[:,1])+0.5)

plt.show()

import pyprind


# ------------------- DEBUT DE L'ALGO -------------------------
#Step 1: Calculating density at each point

# Calcue de la distance
def euclidean_dist (p,q):
    return (np.sqrt((p[0]-q[0])**2 +(p[1]-q[1])**2 ))






# Calcul de la densité
def cal_density (d , dc=0.1):
    n=d.shape[0] # taille de la matrice
    den_arr = np.zeros(n, dtype=np.int) # Definition d'un matrice vide

    for i in range(n):
        for j in range(i+1,n):
            if euclidean_dist(d[i,:], d[j,:]) < dc:
                den_arr[i]+=1
                den_arr[j] += 1

    return (den_arr)

den_arr=cal_density(d[:,0:2], 0.5)

print (max(den_arr)) # la valeurs maximum
#print("den_arr=",den_arr )







#Step 2: Calculating the minimum distance to high density points

def cal_minDist2Peaks(d,den_arr):
    n=d.shape[0]
    mdist2peaks = np.repeat(999,n)
    max_pdist = 0 #store the maximum pairwise distance
    for i in range(n):
        mdist_i=mdist2peaks[i]
        for j in range(i+1,n):
            dist_ij=euclidean_dist(d[i,0:2], d[j,0:2])
            max_pdist=max(max_pdist, dist_ij)
            if den_arr[i] < den_arr[j]:
                mdist_i=min(mdist_i, dist_ij)
            elif den_arr[j] <= den_arr[i]:
                mdist2peaks[j]=min(mdist2peaks[j], dist_ij)
        mdist2peaks[i]=mdist_i

    #update the value for the point with hightest density
    max_den_points= np.argwhere(mdist2peaks == 999)
    print(max_den_points)
    mdist2peaks[max_den_points]=max_pdist
    return (mdist2peaks)

#Finding the number of clusters

mdist2peaks =cal_minDist2Peaks(d,den_arr)

#Plotting the decision graph

def plot_decisionGraph(den_arr,mdist2peaks, thresh=None):

    plt.figure(figsize=(10,8))

    if thresh is not None:
        centroids = np.argwhere((mdist2peaks > thresh) & (den_arr > 1)).flatten()
        noncenter_points = np.argwhere((mdist2peaks < thresh))
    else:
        centroids= None
        noncenter_points=np.arange(den_arr.shape[0])

    plt.scatter(x=den_arr[noncenter_points],
                y=mdist2peaks[noncenter_points],
                color='blue',
                marker='o',
                alpha=0.5,
                s=50)

    if thresh is not None:
        plt.scatter(x=den_arr[centroids],
                    y=mdist2peaks[centroids],
                    color='red',
                    marker='o',
                    alpha=0.6,
                    s=140)

    plt.title('Decision Graph', size=20)
    plt.xlabel(r'$\rho$', size=25)
    plt.ylabel(r'$\delta$', size=25)
    plt.ylim(ymin=min(mdist2peaks-0.5), ymax=max(mdist2peaks+0.5))
    plt.xlim(xmin=min(den_arr - 0.5), xmax=max(den_arr + 0.5))

    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.show()

plot_decisionGraph(den_arr, mdist2peaks,1.7)








# Calcul des centres des classe
#Cluster centroids

thresh=1.0 # hyperparamètres
centroids = np.argwhere((mdist2peaks > thresh) & (den_arr>1)).flatten()

print(centroids.reshape(1, centroids.shape[0]))

plt.figure(figsize=(10,8))

d_centers = d[centroids,:]

for clustId,color in zip(range(0,5),('grey', 'blue', 'red', 'orange', 'brown')):
    plt.scatter(x=d[:,0][d[:,2] == clustId],
                y=d[:,1][d[:,2] == clustId],
                color='grey',
                marker='o',
                alpha=0.4
                )
    # plot the cluster centroids
    plt.scatter(x=d_centers[:,0][d_centers[:,2] == clustId],
                y=d_centers[:,1][d_centers[:,2] == clustId],
                color='purple',
                marker='*',
                s=400,
                alpha=1.0
                )
plt.title('Cluster Centroids', size=20)
plt.xlabel('$x_1$', size=25)
plt.ylabel('$x_2$', size=25)

plt.tick_params(axis='both', which='major', labelsize=18)
plt.xlim(xmin=min(d[:,0])-1, xmax=max(d[:,0])+1)
plt.ylim(ymin=min(d[:,1])-1, ymax=max(d[:,1])+1)

plt.show()


def assign_cluster(df, den_arr, centroids):
    """ Assign points to clusters
    """
    nsize = den_arr.shape[0]
    #print (nsize)
    cmemb = np.ndarray(shape=(nsize,2), dtype='int')
    cmemb[:,:] = -1
    ncm = 0
    for i,cix in enumerate(centroids):
        cmemb[i,0] = cix # centroid index
        cmemb[i,1] = i   # cluster index
        ncm += 1
    da = np.delete(den_arr, centroids)
    inxsort = np.argsort(da)
    for i in range(da.shape[0]-1, -1, -1):
        ix = inxsort[i]
        dist = np.repeat(999.9, ncm)
        for j in range(ncm):
            dist[j] = euclidean_dist(df[ix], df[cmemb[j,0]])
            #print(j, ix, cmemb[j,0], dist[j])
        nearest_nieghb = np.argmin(dist)
        cmemb[ncm,0] = ix
        cmemb[ncm,1] = cmemb[nearest_nieghb, 1]
        ncm += 1
    return(cmemb)

clust_membership = assign_cluster(d, den_arr, centroids)


plt.figure(figsize=(10,8))

for clustId,color in zip(range(0,5),('grey', 'blue', 'red', 'orange', 'brown')):
    cset = clust_membership[clust_membership[:,1] == clustId,0]
    plt.scatter(x=d[cset,0],
                y=d[cset,1],
                color=color,
                marker='o',
                alpha=0.6
                )
plt.title('Clustered Dataset by Density-Peak Algorithm', size=20)
plt.xlabel('$x_1$', size=25)
plt.ylabel('$x_2$', size=25)

plt.tick_params(axis='both', which='major', labelsize=18)
plt.xlim(xmin=np.amin(d[:,0])-1, xmax=np.amax(d[:,0])+1)
plt.ylim(ymin=np.amin(d[:,1])-1, ymax=np.amax(d[:,1])+1)

plt.show()



#from matplotlib import pyplot as plt