from sklearn.datasets import *

covtypes_data_x, covtypes_data_y = load_svmlight_file("covtype.libsvm.binary.scale")

ans_y = [1.0 if y == 2 else -1.0 for y in covtypes_data_y]
ans_x = covtypes_data_x
m = ans_x.shape[0]
data_amt = 100
train_size = int(data_amt)

train_cov_x = ans_x[2*train_size:3*train_size]
train_cov_y = ans_y[2*train_size:3*train_size]

test_cov_x = ans_x[3*train_size+1:6*train_size]
test_cov_y = ans_y[3*train_size+1:6*train_size]

dump_svmlight_file(train_cov_x, train_cov_y, "train/covtype_train.vred.scale")
dump_svmlight_file(test_cov_x, test_cov_y, "test/covtype_test.vred.scale")

print "covtypes done"

#rcv_train
rcv_data_x, rcv_data_y = load_svmlight_file("train/rcv1_train.binary")

ans_y = rcv_data_y
ans_x = rcv_data_x
m = ans_x.shape[0]
data_amt = 40
train_size = int(data_amt)

train_cov_x = ans_x[0:train_size]
train_cov_y = ans_y[0:train_size]

test_cov_x = ans_x[train_size+1:4*train_size]
test_cov_y = ans_y[train_size+1:4*train_size]

dump_svmlight_file(train_cov_x, train_cov_y, "train/rcv1_train.vred", zero_based=False)
dump_svmlight_file(test_cov_x, test_cov_y, "test/rcv1_test.vred", zero_based=False)

print "rcv1 done"

#rcv_train_scaled
rcv_data_x, rcv_data_y = load_svmlight_file("train/rcv1_train.binary.scale")

ans_y = rcv_data_y
ans_x = rcv_data_x
m = ans_x.shape[0]
data_amt = 40
train_size = int(data_amt)

train_cov_x = ans_x[0:train_size]
train_cov_y = ans_y[0:train_size]

test_cov_x = ans_x[train_size+1:4*train_size]
test_cov_y = ans_y[train_size+1:4*train_size]

dump_svmlight_file(train_cov_x, train_cov_y, "train/rcv1_train.vred.scale", zero_based=False)
dump_svmlight_file(test_cov_x, test_cov_y, "test/rcv1_test.vred.scale", zero_based=False)

print "rcv1 scaled done"
