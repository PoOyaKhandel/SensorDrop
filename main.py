from dataset import datasets
from node import Node, CloudNet


# X_train, X_test, Y_train, Y_test = datasets.get_mvmc(te_percent=0.20)
X_train, X_test, Y_train, Y_test = datasets.get_mvmc_concat(te_percent=0.20)

print(X_train.shape)
print(Y_train.shape)

cl = CloudNet()
cl.train_model(X_train, Y_train, bt_s=500, eps=500)
print(cl.eval_model(X_test, Y_test))



