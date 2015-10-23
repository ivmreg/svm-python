from sklearn.datasets import load_svmlight_file
import numpy as np
from random import randrange
from cache import *

class SMO:
    """
    SMO

    **Follows Platt paper**
    """
    def __init__(self, trainpath, testpath):
        self.x_train, self.y_train = load_svmlight_file(trainpath)
        self.xs = self.x_train
        self.num_samp = self.xs.shape[0]
        self.ys = self.y_train
        self.alphas = np.zeros(self.num_samp)
        self.b = 0
        self.w = np.zeros(self.xs.shape[1])
        self.L = 0
        self.c = 0.1
        self.eps = 0.001
        self.H = self.c
        self.point = self.x_train
        self.testpath = testpath
        self.trainpath = trainpath
        self.examineAll = 1
        self.numChanged = 0
        self.E_cache = np.zeros(self.num_samp)
        self.target = self.ys
    
    def stopping_criteria_1(self):
        #cond 1 and cond 2 guaranteed by algo
        #cond1 = np.dot(self.ys, self.alphas) == 0
        #cond2 = 0 <= np.min(self.alphas) and self.c>= np.max(self.alphas)
        cond3 = True
        for i in range(0, self.num_samp):
            ygx = self.target[i] * self.SVM_out(i)
            if self.alphas[i] == 0:
                cond3 = ygx >= 1 - self.eps
            elif self.alphas[i] == self.c:
                cond3 = ygx <= 1 + self.eps
            else:
                cond3 = abs(ygx - 1) < self.eps

            if not cond3:
                break
        return cond3
    
    
    def SVM_out(self, i):
        #xi*self.w + w0, sparse
        return (self.point[i].dot(self.w.T) - self.b)
    
    def SVM_eval(self, point):
        return (point.dot(self.w.T) - self.b)[0, 0]

    @lfu_cache(100000) #caching the last 100000 results
    def kernel(self, i1, i2):
        return self.point[i1].dot(self.point[i2].T)[0, 0]
    
    def takeStep(self, i1, i2):
        if i1 == i2:
            return 0
        alph1 = self.alphas[i1]
        alph2 = self.alphas[i2]
        y1 = self.target[i1]
        y2 = self.target[i2]
        E1 = self.SVM_out(i1) - y1
        E2 = self.SVM_out(i2) - y2
        s = y1 * y2
        if y1 != y2:
            self.L = max(0, alph2 - alph1)
            self.H = min(self.c, self.c + alph2 - alph1)
        else:
            self.L = max(0, alph1 + alph2 - self.c)
            self.H = min(self.c, alph1 + alph2)
        if self.L == self.H:
            return 0

        k11 = self.kernel(i1, i1)
        k12 = self.kernel(i1, i2)
        k22 = self.kernel(i2, i2)
        eta = 2 * k12 - k11 - k22

        if eta < 0:
            a2 = alph2 - (y2 * (E1 - E2) / eta)
            if a2 < self.L: 
                a2 = self.L
            elif a2 > self.H:
                a2 = self.H
        else:
            v2 = self.SVM_out(i2) + self.b - y1 * alph1 * self.kernel(i1, i2) - y2 * alph2 * self.kernel(i2, i2) #eq 12.21 
            L_H_diff = (self.L - self.H) - 0.5 * self.kernel(i2, i2) * (self.L * self.L - self.H * self.H) - s * self.kernel(i1, i2) * alph1 * (self.L - self.H) - y2 * (self.L - self.H) * v2   #diff b/w objective at a2 = self.L and a2 = H, ref 12.19
            if L_H_diff > self.eps:
                a2 = self.L
            elif L_H_diff < -self.eps:
                a2 = self.H
            else:
                a2 = alph2
        ###
        ##arbit bounds checking, as done in paper pseudocode
        if a2 < 1e-8:
            a2 = 0
        elif a2 > self.c - 1e-8:
            a2 = self.c
        ##still more bounds checks
        if abs(a2 - alph2) < self.eps * (a2 + alph2 + self.eps):
            return 0
        a1 = alph1 + s*(alph2-a2)
        #update threshold, eqns 12.9 and 12.10
        b1 = E1 + y1 * (a1 - alph1) * self.kernel(i1, i1) + y2 * (a2 - alph2) * self.kernel(i1, i2) + self.b
        b2 = E2 + y1 * (a1 - alph1) * self.kernel(i1, i2) + y2 * (a2 - alph2) * self.kernel(i2, i2) + self.b
        bnew = (b1 + b2)/2

        self.w = self.w + (y1 * (a1 - alph1) * self.point[i1]) + (y2 * (a2 - alph2) * self.point[i2]) #eqn 12.12

        for k in set(range(len(self.E_cache))) - set([i1, i2]) :
            self.E_cache[k] = self.E_cache[k] + y1 * (a1 - alph1) * self.kernel(i1, k) + y2 * (a2 - alph2) * self.kernel(i2, k) + (bnew - self.b) #eqn 12.11

        self.b = bnew
        self.alphas[i1] = a1
        self.alphas[i2] = a2
        return 1
    
    def examineExample(self, i2):
        y2 = self.target[i2]
        alph2 = self.alphas[i2]
        E2 = self.SVM_out(i2) - self.target[i2]
        r2 = E2 * y2
        if (r2 < -self.eps and alph2 < self.c) or (r2 > self.eps and alph2 > 0):
            if len(self.alphas.nonzero()) + sum( self.alphas == self.c ) > 1:
                i1 = np.argmax(np.abs(E2 - self.E_cache)) # second choice heuristic
                if self.takeStep(i1, i2):
                    return 1

            non_zero_non_c = np.where((self.alphas < self.c) & (self.alphas > 0))[0]
            if len(non_zero_non_c) != 0:
                spl = randrange(len(non_zero_non_c))
                for ind in range(spl,len(non_zero_non_c)) + range(0,spl):
                    i1 = non_zero_non_c[ind]
                    if self.takeStep(i1, i2):
                        return 1

            spl = randrange(len(self.alphas))
            for i1 in range(spl, len(self.alphas)) + range(0, spl):
                if self.takeStep(i1, i2):
                    return 1
        return 0
    
        
    def train(self):
        it = 0
        while (self.examineAll or self.numChanged > 0) and not self.stopping_criteria_1():
            it += 1
            self.numChanged = 0
            if self.examineAll:
                for I in range(self.num_samp):
                    self.numChanged += self.examineExample(I)
            else:
                for I in np.where((self.alphas != 0) & (self.alphas != self.c))[0]:
                    self.numChanged += self.examineExample(I)

            if self.examineAll == 1:
                self.examineAll = 0
            elif self.numChanged == 0:
                self.examineAll = 1
            #output is w and b
            print it
