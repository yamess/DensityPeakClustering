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

