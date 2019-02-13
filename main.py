from dataset import datasets
from node import Node, CloudNet
from rl import PolicyNetwork, RL
import numpy as np
from sklearn.metrics import accuracy_score as acc


X_train, X_test, Y_train, Y_test = datasets.get_mvmc(te_percent=0.20)
# X_train, X_test, Y_train, Y_test = datasets.get_mvmc_concat(te_percent=0.20)

print(X_train['0'].shape)
print(X_test['0'].shape)


# exit()
#  train the model
cl = CloudNet(train=1)

iftrain_CloudNet=0
iftrain_RLNet =0

if iftrain_CloudNet==1:
    cl.train_model(X_train, Y_train, bt_s=50, eps=20)

# instantiate node and cloud network
node = []
for i in range(6):
    node.append(Node(i))

# exit()


if iftrain_RLNet==1:

    node_output = []
    for l in range(6):
        print("input node", l, "is processing.")
        node_output.append(node[l].calculate(X_train[str(l)]))

    rl = RL()
    rl.train(node_output, Y_train, 20)



X_test=X_train
Y_test=Y_train

node_output = []
for l in range(6):
    node_output.append(node[l].calculate(X_test[str(l)]))

cl = CloudNet(train=0)

print("here1")
# print(cl.eval_model(node_output, Y_test))
prediction_res = cl.model.pred(node_output)
pred_res = []
for res in prediction_res:
    pred_res.append([np.argmax(res)])

print(acc(pred_res, Y_test))
print("here2")


policy_net = PolicyNetwork(train=0)
print("here3")
policy_for_test = policy_net.feed(node_output)
policy_for_test = policy_for_test * 0.9 + 0.1 * (1 - policy_for_test)
u = policy_for_test > np.random.random(policy_for_test.shape)
u = u * 1.0
# policy_for_test[policy_for_test<0.5]=0
# print(policy_for_test.shape)
print(np.count_nonzero(u, axis=0))
print("here4")

cl = CloudNet(0)
y = cl.calculate(node_output, u)
y = np.argmax(y, axis=1)

print(y.shape)
print(acc(y, Y_test))


