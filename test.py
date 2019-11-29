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


d = [[6, 8, 0.47, [9, 17, 18, 53, 73, 96, 137, 173]], [0, 7, 0.45, [15, 43, 45, 57, 139, 151, 191]], [15, 7, 0.45, [21, 57, 88, 146, 152, 169, 177]], [19, 7, 0.48, [35, 75, 90, 91, 117, 156, 183]], [9, 6, 0.49, [17, 53, 73, 77, 100, 137]], [10, 5, 0.46, [46, 62, 108, 136, 164]], [12, 5, 0.5, [75, 156, 166, 194, 199]], [2, 5, 0.44, [28, 99, 104, 123, 158]], [4, 5, 0.4, [15, 57, 152, 169, 193]], [27, 5, 0.48, [60, 78, 128, 144, 168]], [43, 5, 0.46, [45, 57, 151, 171, 191]], [46, 5, 0.5, [62, 108, 131, 136, 164]], [49, 5, 0.41, [94, 111, 112, 167, 179]], [94, 5, 0.47, [111, 112, 130, 153, 179]], [23, 4, 0.48, [24, 92, 113, 180]], [45, 4, 0.49, [139, 151, 171, 191]], [24, 4, 0.45, [92, 133, 135, 184]], [17, 4, 0.39, [18, 53, 73, 96]], [51, 4, 0.47, [159, 185, 196, 198]], [53, 4, 0.47, [73, 77, 100, 137]], [55, 4, 0.45, [66, 71, 103, 130]], [35, 4, 0.47, [75, 90, 156, 183]], [146, 4, 0.46, [152, 169, 177, 197]], [21, 3, 0.46, [88, 139, 146]], [54, 3, 0.5, [56, 150, 186]], [30, 3, 0.47, [31, 91, 117]], [56, 3, 0.48, [138, 150, 186]], [57, 3, 0.43, [152, 169, 193]], [60, 3, 0.37, [78, 128, 168]], [61, 3, 0.45, [105, 143, 174]], [70, 3, 0.48, [97, 100, 132]], [77, 3, 0.46, [100, 137, 144]], [80, 3, 0.34, [106, 111, 153]], [82, 3, 0.46, [90, 150, 183]], [88, 3, 0.38, [146, 177, 197]], [92, 3, 0.39, [133, 180, 184]], [7, 3, 0.42, [67, 141, 176]], [111, 3, 0.48, [112, 147, 153]], [8, 3, 0.37, [20, 47, 50]], [152, 3, 0.44, [169, 177, 193]], [14, 2, 0.26, [40, 197]], [62, 2, 0.27, [81, 136]], [65, 2, 0.2, [140, 142]], [66, 2, 0.36, [71, 130]], [67, 2, 0.39, [141, 176]], [18, 2, 0.13, [73, 96]], [71, 2, 0.48, [121, 130]], [73, 2, 0.45, [96, 173]], [74, 2, 0.43, [108, 131]], [75, 2, 0.49, [90, 156]], [25, 2, 0.49, [158, 160]], [5, 2, 0.28, [26, 122]], [28, 2, 0.47, [123, 158]], [86, 2, 0.42, [162, 181]], [16, 2, 0.32, [70, 97]], [90, 2, 0.46, [91, 183]], [22, 2, 0.48, [63, 147]], [93, 2, 0.42, [118, 135]], [37, 2, 0.34, [44, 83]], [99, 2, 0.34, [104, 123]], [103, 2, 0.49, [130, 178]], [104, 2, 0.47, [123, 158]], [108, 2, 0.49, [131, 164]], [38, 2, 0.29, [165, 189]], [112, 2, 0.36, [167, 179]], [134, 2, 0.48, [142, 186]], [40, 2, 0.47, [143, 197]], [151, 2, 0.25, [171, 191]], [42, 2, 0.13, [114, 187]], [159, 2, 0.47, [185, 196]], [165, 2, 0.49, [172, 189]], [166, 2, 0.46, [194, 199]], [167, 2, 0.48, [178, 179]], [169, 2, 0.47, [177, 193]], [64, 1, 0.33, [145]], [11, 1, 0.33, [68]], [91, 1, 0.14, [117]], [29, 1, 0.29, [34]], [39, 1, 0.18, [41]], [69, 1, 0.49, [155]], [97, 1, 0.45, [137]], [26, 1, 0.27, [122]], [100, 1, 0.22, [137]], [31, 1, 0.47, [156]], [72, 1, 0.16, [95]], [105, 1, 0.44, [110]], [106, 1, 0.24, [153]], [32, 1, 0.39, [190]], [109, 1, 0.4, [171]], [44, 1, 0.28, [83]], [59, 1, 0.29, [84]], [114, 1, 0.14, [187]], [121, 1, 0.25, [181]], [122, 1, 0.39, [182]], [124, 1, 0.2, [148]], [125, 1, 0.17, [161]], [126, 1, 0.49, [145]], [127, 1, 0.37, [149]], [133, 1, 0.18, [184]], [76, 1, 0.42, [159]], [136, 1, 0.38, [164]], [138, 1, 0.17, [150]], [139, 1, 0.47, [191]], [140, 1, 0.35, [142]], [143, 1, 0.18, [174]], [144, 1, 0.35, [168]], [33, 1, 0.13, [69]], [78, 1, 0.28, [128]], [79, 1, 0.33, [164]], [156, 1, 0.44, [194]], [158, 1, 0.35, [160]], [13, 1, 0.33, [154]], [161, 1, 0.42, [176]], [162, 1, 0.44, [178]], [81, 1, 0.44, [136]], [47, 1, 0.34, [50]], [63, 1, 0.35, [127]], [87, 1, 0.16, [107]], [171, 1, 0.47, [191]], [172, 1, 0.49, [189]], [177, 1, 0.28, [197]], [185, 1, 0.07, [196]], [48, 0, 0.0, []], [123, 0, 0.0, []], [68, 0, 0.0, []], [95, 0, 0.0, []], [96, 0, 0.0, []], [58, 0, 0.0, []], [128, 0, 0.0, []], [129, 0, 0.0, []], [130, 0, 0.0, []], [131, 0, 0.0, []], [132, 0, 0.0, []], [98, 0, 0.0, []], [36, 0, 0.0, []], [135, 0, 0.0, []], [50, 0, 0.0, []], [137, 0, 0.0, []], [101, 0, 0.0, []], [102, 0, 0.0, []], [83, 0, 0.0, []], [141, 0, 0.0, []], [142, 0, 0.0, []], [84, 0, 0.0, []], [85, 0, 0.0, []], [145, 0, 0.0, []], [20, 0, 0.0, []], [147, 0, 0.0, []], [148, 0, 0.0, []], [149, 0, 0.0, []], [150, 0, 0.0, []], [107, 0, 0.0, []], [52, 0, 0.0, []], [153, 0, 0.0, []], [154, 0, 0.0, []], [155, 0, 0.0, []], [1, 0, 0.0, []], [157, 0, 0.0, []], [110, 0, 0.0, []], [89, 0, 0.0, []], [160, 0, 0.0, []], [34, 0, 0.0, []], [113, 0, 0.0, []], [163, 0, 0.0, []], [164, 0, 0.0, []], [3, 0, 0.0, []], [115, 0, 0.0, []], [116, 0, 0.0, []], [168, 0, 0.0, []], [117, 0, 0.0, []], [170, 0, 0.0, []], [118, 0, 0.0, []], [119, 0, 0.0, []], [173, 0, 0.0, []], [174, 0, 0.0, []], [175, 0, 0.0, []], [176, 0, 0.0, []], [120, 0, 0.0, []], [178, 0, 0.0, []], [179, 0, 0.0, []], [180, 0, 0.0, []], [181, 0, 0.0, []], [182, 0, 0.0, []], [183, 0, 0.0, []], [184, 0, 0.0, []], [41, 0, 0.0, []], [186, 0, 0.0, []], [187, 0, 0.0, []], [188, 0, 0.0, []], [189, 0, 0.0, []], [190, 0, 0.0, []], [191, 0, 0.0, []], [192, 0, 0.0, []], [193, 0, 0.0, []], [194, 0, 0.0, []], [195, 0, 0.0, []], [196, 0, 0.0, []], [197, 0, 0.0, []], [198, 0, 0.0, []], [199, 0, 0.0, []]]
t = np.transpose(d)