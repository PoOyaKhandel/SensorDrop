from dataset import datasets
from node import Node, CloudNet
from rl import PolicyNetwork, RL
import numpy as np
from sklearn.metrics import accuracy_score as acc


X_train, X_test, Y_train, Y_test = datasets.get_mvmc(te_percent=0.20)
# X_train, X_test, Y_train, Y_test = datasets.get_mvmc_concat(te_percent=0.20)


#  train the model
cl = CloudNet(train=1)
cl.train_model(X_train, Y_train, bt_s=100, eps=2)

# instantiate node and cloud network
node = []
for i in range(6):
    node.append(Node(i))



# exit()

node_output = []
for l in range(6):
    node_output.append(node[l].calculate(X_train[str(l)]))

rl = RL()
rl.train(node_output, Y_train, 10)


node_output = []
for l in range(6):
    node_output.append(node[l].calculate(X_test[str(l)]))

cl = CloudNet(0)
print(cl.eval_model(node_output, Y_test))


policy_net = PolicyNetwork(train=0)
policy_for_test = policy_net.feed(node_output)
print(np.count_nonzero(policy_for_test, axis=0))

cl = CloudNet(0)
y = cl.calculate(node_output, policy_for_test)
y = np.argmax(y, axis=1)

print(y.shape)
print(acc(y, Y_test))


