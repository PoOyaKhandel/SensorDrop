from dataset import datasets
from node import  CloudNet
from rl import PolicyNetwork, RL
import numpy as np
from sklearn.metrics import accuracy_score as acc
import tensorflow as tf
from keras.utils.vis_utils import plot_model
import os

# os.environ["PATH"] += os.pathsep + 'C:\Users\vpour\AppData\Local\conda\conda\envs\tensorflow\Lib\site-packages\graphviz'

ses = tf.InteractiveSession()
# ses.run(tf.global_variables_initializer())

X_train, X_test, Y_train, Y_test = datasets.get_mvmc(te_percent=0.20)
# X_train, X_test, Y_train, Y_test = datasets.get_mvmc_concat(te_percent=0.20)

print(X_train['0'].shape)
print(X_test['0'].shape)


# exit()
#  train the model

iftrain_CloudNet=1
iftrain_RLNet =1
iftest_compl = 1

if iftrain_CloudNet==1:
    cl_t = CloudNet(train=1)
    cl_t.train_model(X_train, Y_train, bt_s=50, eps=30)

# instantiate node and cloud network
# node = []
# for i in range(6):
#     node.append(Node(i))
# # exit()
# cl_t = CloudNet(train=1)
# plot_model(cl_t.model, to_file='all.png')
# plot_model(cl_t.node[1].model, to_file='node.png')
# plot_model(cl_t.model_cloud, to_file='cloud.png')

# exit()

if iftrain_RLNet==1:
    
    cl_rl = CloudNet(train=0)

    node_output = []
    for l in range(6):
        print("input node", l, "is processing.")
        node_output.append(cl_rl.node[l].predict(X_train[str(l)].reshape((-1, 32, 32, 3))))

    rl = RL(cl_rl, train=1)
    # rl.train(node_output, Y_train, 60,ses)
    rl.train(node_output, Y_train, 60,ses)


# exit()

# node_output = []
# for l in range(6):
#     node_output.append(node[l].calculate(X_test[str(l)]))
if iftest_compl==1:
    cl_t = CloudNet(train=0)

    X_test=X_train
    Y_test=Y_train

    print("here1")
    # print(cl.eval_cloud_model(node_output, Y_test))
    x = []
    for l in range(6):
        x.append(X_test[str(l)].reshape((-1, 32, 32, 3)))
    print(cl_t.eval_comp_model(x, Y_test))
    print("here2")

    node_output = []
    for l in range(6):
        print("Test --input node", l, "is processing.")
        node_output.append(cl_t.node[l].predict(X_train[str(l)].reshape((-1, 32, 32, 3))))
    print(cl_t.eval_cloud_model([node_output], Y_test))
    print("here2.5")


    # policy_net = PolicyNetwork(train=0)

    # print("here3")
    # policy_for_test = policy_net.feed(node_output_t)
    # print(policy_for_test.shape)
    # print(np.count_nonzero(policy_for_test, axis=0))
    # print("here4")

    # y = cl.calculate_claud(node_output, policy_for_test)
    # y = np.argmax(y, axis=1)

    # print(y.shape)
    # print(acc(y, Y_test))


