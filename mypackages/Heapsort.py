# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 22:27:50 2019

@author: Prof Gilles Caporossi
"""
import array


class Heap:
    def __init__(self):
        self.data = array.array('i')
        self.data.append(0)
        self.nb_elts = 0

    def add(self, i):
        for i in range(n):
            self.data.append(i)
            self.nb_elts += 1
            pos = self.nb_elts
            while pos > 1:
                pos_pere = pos // 2
                if self.data[pos] < self.data[pos_pere]:
                    self.data[pos], self.data[pos_pere] = self.data[pos_pere],self.data[pos]
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

    def not_empty(self):
        if self.nb_elts >= 1:
            return True
        else:
            return False

    def sort(self, arr):
        for i in arr:
            self.add(i)
        result = []
        while self.not_empty():
            result.append(self.getmin())
        return result


liste_a_trier = [3, 6, 2, 8, 5, 4, 3, 2, 1, 9]

h = Heap()
print(h.sort(liste_a_trier))

