from dataset import datasets
from node import Node, CloudNet


X_train, X_test, Y_train, Y_test = datasets.get_mvmc(te_percent=0.20)


cl = CloudNet()
cl.train_model(X_train[0], Y_train[0], 100, 5)
