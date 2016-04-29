from sklearn.datasets import load_svmlight_file
from scipy.spatial.distance import *
from sklearn.metrics.pairwise import *
import numpy as np
from cvxopt import matrix, solvers
from cache import *
import random

class Chunking:
    def __init__(self, trainpath, testpath):
        self.trainpath = trainpath
        self.testpath = testpath
        self.x_train, self.y_train = load_svmlight_file(trainpath)
        self.c = 1
        self.eps = 0.001
        self.num_samp = self.x_train.shape[0]
        self.width = self.x_train.shape[1]
        self.alphas = np.zeros(self.num_samp)
        self.yixi = [a*b for a,b in zip(self.y_train, self.x_train)]
        self.chunk_bound = 10 ** 7 #problem is happen #otherwise O(n^3) becomes infeasible
        self.w = np.zeros(self.x_train.shape[1])
        self.b = 0

    def grad_alph(self, i):
        return  self.y_train[i] * self.x_train[i].dot(self.w.T).toarray()[0][0] - 1
    
    def get_accu(self):
        """
        Run on test data
        """
        x_test, y_test = load_svmlight_file(self.testpath)
        corr = 0
        for i in range(x_test.shape[0]):
            if np.sign(self.SVM_eval(x_test[i])) == y_test[i]:
                corr +=1
        
        return corr * 1.0/x_test.shape[0]

    def stopping_criteria(self, ty=1):
        if ty == 1:
            return self.Si == self.old_Si
        elif ty==3:
            l_val = 0
            u_val = 10 ** 10
            for i in range(len(self.alphas)):
                if (self.alphas[i] < self.c and self.y_train[i] == 1) or (self.alphas[i] == self.c) :
                    l_val = max(self.b - self.y_train[i] * self.grad_alph(i), l_val)
                else:
                    u_val = min(-self.y_train[i] * self.grad_alph(i) + self.b, u_val)
            return (u_val - l_val + self.eps >= 0)
        elif ty==2:
            term1 = self.w.dot(self.w.T).toarray()[0][0] - np.sum(self.alphas)
            term2 = 0
            for i in range(len(self.alphas)):
                xii = 1 - self.y_train[i] * self.SVM_out(i)
                term2 += xii
            return abs(term1 + self.c * term2) <= self.eps
        return True

    def solve_with_Si(self):
        """
        Solve the QP with Si's as the indices
        and put it in self.alphas
        """
        P = np.zeros((len(self.Si), len(self.Si)))
        for i in range(len(self.Si)):
            for j in range(len(self.Si)):
                P[i][j] = self.y_train[i] * self.y_train[j] * self.kernel(i, j)
        q = -1 * np.ones(len(self.Si))
        G = np.append(np.eye(len(P[0]), len(P)), -1 * np.eye(len(P[0]),len(P)), axis=0)
        h = np.append((self.c * np.ones(len(P))), np.zeros(len(P)))
        A = self.y_train[self.Si]
        b = 0.0

        soln = solvers.qp(matrix(P), matrix(q), matrix(G), matrix(h), matrix(A).T, matrix(b))['x']    
        for i in range(len(self.Si)):
            self.alphas[self.Si[i]] = soln[i]
        self.update_weights()
    
    def update_weights(self):
        self.w = np.sum([a*b for a,b in zip(self.alphas, self.yixi)], axis=0)
        ind = -1
        #below loop could be done in single step
        for i in range(len(self.alphas)):
            if self.alphas[i] > 0 and self.alphas[i] < self.c:
                ind = i
                break
        self.b = self.y_train[ind] - self.x_train[ind].dot(self.w.T)[0, 0]

    def SVM_out(self, i):
        return (self.x_train[i].dot(self.w.T)[0, 0] + self.b)

    def SVM_eval(self, point):
        return (point.dot(self.w.T)[0, 0] + self.b)

    @lru_cache(100000) #caching the last 100000 results
    def kernel(self, i1, i2):
        return self.x_train[i1].dot(self.x_train[i2].T)[0, 0]
    
    def most_violating(self):
        avail_set = set(range(self.num_samp)) - set(self.Si)
        len_so_far = len(self.Si)
        chosen_set = set()
	candidate_list = []
        for i in avail_set:
            if self.y_train[i] * self.SVM_out(i) - 1 < 0:
                candidate_list.append(i)

        if len(candidate_list) == 0:
            return candidate_list
        num_to_take = int(0.3 * len(candidate_list))
        return set(np.take(candidate_list, range(max(num_to_take, 1))))
        """
        for i in avail_set:
	    val = self.y_train[i] * self.SVM_out(i) - 1 
            if val < 0 :
                candidate_list.append((val, i))
	candidate_list.sort()
	
	sel_part = max(int(0.1 * len(candidate_list)), 1)
	#choose 10% of the candidates

        if len(chosen_set) + len_so_far > self.chunk_bound:
            return random.sample(chosen_set, (self.chunk_bound - len_so_far)/2 )
        else:
            if sel_part > len(candidate_list):
                print candidate_list, sel_part
                return set()
            for i in range(sel_part):
                print candidate_list, i
                chosen_set.add(candidate_list[i][1])
        return chosen_set
        """
    def train(self, crit):
        i = 1
        self.Si = list(np.random.choice([0, 1], size=self.num_samp, p=[0.95, 0.05]).nonzero()[0]) # only 5% expected to be chosen
        if len(self.Si) > self.chunk_bound:
            self.Si = random.sample(self.Si, self.chunk_bound/2)
        if len(self.Si) == 0:
            self.Si = range(int(0.1 * self.num_samp))

	self.old_Si = set()

        self.solve_with_Si()
        while not self.stopping_criteria(crit):
            i +=1
	    print "iteration:", i
            if crit!=1 and self.stopping_criteria(1):
                print "Stopping criteria 1 already satisfied"
                break
            most_violating_set = self.most_violating()
	    self.old_Si = self.Si
            self.Si = list(set(self.Si).union(most_violating_set))#- set(np.where(self.alphas == 0)[0]))
            self.solve_with_Si()


