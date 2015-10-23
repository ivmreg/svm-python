#!/bin/zsh

../lib/libsvm-3.20/svm-scale -l 0 -u 1 train/leu > train/leu.scale
../lib/libsvm-3.20/svm-scale -l 0 -u 1 train/rcv1_train.binary > train/rcv1_train.binary.scale

../lib/libsvm-3.20/svm-scale -l 0 -u 1 test/leu.t > test/leu.t.scale
../lib/libsvm-3.20/svm-scale -l 0 -u 1 test/rcv1_test.binary > test/rcv1_test.binary.scale
