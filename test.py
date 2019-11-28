from copy import deepcopy
import numpy as np


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
    ind_tmp = deepcopy(np.arange(n))
    if n_to_sort == 1:
        res = deepcopy(arr_tmp[0])
        if how == 'max':
            for j in range(n_to_sort, n):
                if arr_tmp[j] > res:
                    res,arr_tmp[j] = arr_tmp[j],res
        elif how == 'min':
            for j in range(n_to_sort,n):
                if arr_tmp[j] < res:
                    res, arr_tmp[j] = arr_tmp[j], res
        return res
    else:
        res = deepcopy(arr_tmp[:n_to_sort])
        table = [[v, k] for k, v in enumerate(arr_tmp)]
        ind = np.arange(n-n_to_sort)
        resultat = []
        swap = False
        if how == 'max':
            for i in range(n_to_sort):
                tmp = deepcopy(res[i])
                for j in range(n_to_sort,n):
                    if arr_tmp[j] > tmp:
                        tmp,arr_tmp[j] = arr_tmp[j],tmp
                        swap = True
                if swap:
                    res[i] = tmp
                    swap = False
            return res



a = [0,1,21,10, 3,4,5,90,24,56,45]
# b = [[k, i] for i, k in enumerate(a[:4])]
print(sort_n(a, n_to_sort=2, how='max'))

#  Fonction de tri n max
def tri_max(arr, n_to_sort=1, how='max'):
    # Partie commune
    from copy import deepcopy
    import numpy as np
    data_val = deepcopy(arr)
    n = len(data_val)
    data_id = list(np.arange(n))
    data = np.transpose([data_val, data_id])
    result = data[:n_to_sort]
    for i in range(n_to_sort):
        tmp_val = result[i, 0]
        tmp_id = result[i, 1]
        swap = False
        for j in range(n_to_sort, n):
            if data[j, 0] > tmp_val:
                data[j, 0], tmp_val = tmp_val, data[j, 0]  # echanger les valeurs
                data[j, 1], tmp_id = tmp_id, data[j, 1]  # echanger les id
                swap = True
            if swap:
                result[i, 0] = tmp_val
                result[i, 1] = tmp_id
                swap = False
    return result


def tri_min(arr, n_to_sort=1, how='max'):
    from copy import deepcopy
    import numpy as np
    data_val = deepcopy(arr)
    n = len(data_val)
    data_id = list(np.arange(n))
    data = np.transpose([data_val, data_id])
    result = data[:n_to_sort]
    for i in range(n_to_sort):
        tmp_val = result[i, 0]
        tmp_id = result[i, 1]
        swap = False
        for j in range(n_to_sort, n):
            if data[j, 0] < tmp_val:
                data[j, 0], tmp_val = tmp_val, data[j, 0]  # echanger les valeurs
                data[j, 1], tmp_id = tmp_id, data[j, 1]  # echanger les id
                swap = True
            if swap:
                result[i, 0] = tmp_val
                result[i, 1] = tmp_id
                swap = False
    return result


def tri(arr, n_to_sort=1, how='max'):
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
                if data[j,0] < tmp_val:
                    data[j,0],tmp_val = tmp_val,data[j,0]  # echanger les valeurs
                    data[j,1],tmp_id = tmp_id,data[j,1]  # echanger les id
                    swap = True
                if swap:
                    result[i,0] = tmp_val
                    result[i,1] = tmp_id
                    swap = False
        return result
    elif str(how).upper() == 'MAX':
        for i in range(n_to_sort):
            tmp_val = result[i,0]
            tmp_id = result[i, 1]
            swap = False
            for j in range(n_to_sort,n):
                if data[j,0] > tmp_val:
                    data[j, 0], tmp_val = tmp_val, data[j, 0]  # echanger les valeurs
                    data[j, 1], tmp_id = tmp_id, data[j, 1]  # echanger les id
                    swap = True
                if swap:
                    result[i,0] = tmp_val
                    result[i,1] = tmp_id
                    swap = False
        return result
    else:
        print("how prend uniquement les valeurs 'min' ou 'max'")




import numpy as np
#test_data = np.transpose([[12,34,5,9,90,435,6,7,8],[0,3,5,9,10,35,4,7,8]])
d = [12,34,5,9,90,435,6,5,7,8]
print(tri(d, 5, how='Max'))

print(d[1][1])

# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 22:27:50 2019

@author: 00009701
"""
import array


class Heap:
    def __init__(self):
        self.data = array.array('i')
        self.data.append(0)
        self.nb_elts = 0

    def add(self,i):
        self.data.append(i)
        self.nb_elts += 1
        pos = self.nb_elts
        while pos > 1:
            pos_pere = pos // 2
            if self.data[pos] < self.data[pos_pere]:
                self.data[pos],self.data[pos_pere] = self.data[pos_pere],self.data[pos]
                pos = pos_pere
            else:
                pos = 1

    def getmin(self):
        minimum = self.data[1]
        pos = 1
        while pos <= self.nb_elts:  # modifie pour <= au lieu de <
            posfils1 = pos * 2
            posfils2 = pos * 2 + 1
            if posfils1 <= self.nb_elts:  # on a au moins 1 fils
                if posfils2 <= self.nb_elts:  # on a 2 fils!!
                    if self.data[posfils1] < self.data[posfils2]:
                        self.data[pos] = self.data[posfils1]
                        pos = posfils1
                    else:
                        self.data[pos] = self.data[posfils2]
                        pos = posfils2
                else:  # on a un seul fils
                    self.data[pos] = self.data[posfils1]
                    pos = posfils1
            else:  # on est en bas
                if pos < self.nb_elts:
                    self.data[pos] = self.data[self.nb_elts]
                    while pos > 1:
                        pos_pere = pos // 2
                        if self.data[pos] < self.data[pos_pere]:
                            self.data[pos],self.data[pos_pere] = self.data[pos_pere],self.data[pos]
                            pos = pos_pere
                        else:
                            pos = 1
                self.data.pop()
                self.nb_elts -= 1
                return minimum

    def notEmpty(self):
        if self.nb_elts >= 1:
            return True
        else:
            return False


liste_a_trier = [3,6,2,8,5,4,3,2,1,9]

h = Heap()
for i in liste_a_trier:
    h.add(i)

liste_triee = []
while h.notEmpty():
    liste_triee.append(h.getmin())

print(liste_triee)
