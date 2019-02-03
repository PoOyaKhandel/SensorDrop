from dataset import datasets
from node import Node, CloudNet
from rl import PolicyNetwork, RL


# def train_policy(node_out, cl, policy_net, epoch=5):

#     for ep in range(epoch):
#         u = policy_net.calculate(node_out)
#         cloud_input = [node_out[i] for _ in range(6) if u[i] == 1]
#         cloud_output = cl.calculate(cloud_input)
#         cloud_prediction = True  ##### This line should be edited, we need something to check cloud prediction
#         policy_net.train(node_out, u, cloud_output)


X_train, X_test, Y_train, Y_test = datasets.get_mvmc(te_percent=0.20)
# X_train, X_test, Y_train, Y_test = datasets.get_mvmc_concat(te_percent=0.20)


#  train the model
cl = CloudNet(train=1)
cl.train_model(X_train, Y_train, bt_s=100, eps=2)

# instantiate node and cloud network
node = []
for i in range(6):
    node.append(Node(i))

node_output = []
for l in range(6):
    node_output.append(node[l].calculate(X_train[str(l)]))

rl = RL()
rl.train(node_output, Y_train, 10)


