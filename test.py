from copy import deepcopy
import numpy as np


def sort_n(arr,n_to_sort=1,how='max'):
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
            for j in range(n_to_sort,n):
                if arr_tmp[j] > res:
                    res,arr_tmp[j] = arr_tmp[j],res
        elif how == 'min':
            for j in range(n_to_sort,n):
                if arr_tmp[j] < res:
                    res, arr_tmp[j] = arr_tmp[j], res
        return res
    else:
        res = deepcopy(arr_tmp[:n_to_sort])
        ind = deepcopy(ind_tmp[:n_to_sort])
        swap = False
        if how == 'max':
            for i in range(n_to_sort):
                tmp_val = deepcopy(res[i])
                tmp_ind = deepcopy(ind[i])
                for j in range(n_to_sort,n):
                    if arr_tmp[j] > tmp_val:
                        tmp_val, arr_tmp[j] = arr_tmp[j], tmp_val
                        tmp_ind,ind_tmp[j] = ind_tmp[j],tmp_ind
                        ind, ind_tmp[j] = ind_tmp[j], ind
                        swap = True
                if swap:
                    res[i] = tmp_val
                    ind[i] = ind
                    swap = False
            return res, ind
        elif how == 'min':
            for i in range(n_to_sort):
                tmp = deepcopy(arr_tmp[i])
                for j in range(n_to_sort, n):
                    if arr_tmp[j] < tmp:
                        tmp,arr_tmp[j] = arr_tmp[j],tmp
                        swap = True
                if swap:
                    res[i] = tmp
                    # resultat.append(tmp)
                    swap = False
            return res


a = [0,1,21,10, 3,4,5,90,24,56,45]
# b = [[k, i] for i, k in enumerate(a[:4])]
print(sort_n(a, n_to_sort=2, how='max'))