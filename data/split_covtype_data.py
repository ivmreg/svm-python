from sklearn.datasets import *

covtypes_data_x, covtypes_data_y = load_svmlight_file("covtype.libsvm.binary.scale")

ans_y = [1.0 if y == 2 else -1.0 for y in covtypes_data_y]
ans_x = covtypes_data_x.toarray()
m = len(ans_x)
train_size = int(0.2*m)

train_cov_x = ans_x[0:train_size]
train_cov_y = ans_y[0:train_size]

test_cov_x = ans_x[train_size+1:m]
test_cov_y = ans_y[train_size+1:m]

dump_svmlight_file(train_cov_x, train_cov_y, "train/covtype_train.scaled", zero_based=False)
dump_svmlight_file(test_cov_x, test_cov_y, "test/covtype_test.scaled", zero_based=False)

