#!/bin/python
from smo import SMO
import cPickle as pickle
import time
import sys

if len(sys.argv) == 5:
    args = sys.argv
    A = SMO.SMO(args[2], args[3])
    start = time.clock()
    print "Training on", args[2]
    A.train(int(args[1]))
    stop = time.clock()
    print "Time taken to train:", str(stop - start)
    print "Accuracy: ", A.get_accu()
    with open( args[4] , "w+") as f:
        pickle.dump(A, f, pickle.HIGHEST_PROTOCOL)
else:
    print "Usage: ./script_smo.py <stopping_criterion> <train_data> <test_data> <model_output_file>"
    print "Outputs the info about the training"
