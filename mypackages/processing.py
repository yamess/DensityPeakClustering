def sort_n(arr, n_to_sort=1, how='max'):
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