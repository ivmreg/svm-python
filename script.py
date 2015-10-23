#!/bin/python
from smo import SMO
import cPickle as pickle
A = SMO.SMO("data/train/rcv1_train.red.scale", "data/test/rcv1_test.red.scale")
A.train()
with open( "smo_models/rcv1_scale_model" , "w+") as f:
    pickle.dump(A, f, pickle.HIGHEST_PROTOCOL)
