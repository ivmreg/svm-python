#!/bin/zsh

../lib/libsvm-3.20/svm-scale -l 0 -u 1 -s train/leu.params train/leu > train/leu.scale
../lib/libsvm-3.20/svm-scale -l 0 -u 1 -s train/rcv1.params train/rcv1_train.binary > train/rcv1_train.binary.scale

echo "Train scaled"

../lib/libsvm-3.20/svm-scale -l 0 -u 1 -r train/leu.params test/leu.t > test/leu.t.scale
../lib/libsvm-3.20/svm-scale -l 0 -u 1 -r train/rcv1.params test/rcv1_test.binary > test/rcv1_test.binary.scale

echo "Finished"
