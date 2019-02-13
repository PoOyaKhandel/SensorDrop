from node import Node
from cloudnet import CloudNet
from dataset import datasets
import tensorflow as tf
from sklearn.metrics import accuracy_score as acc
import numpy as np

X_train, X_test, Y_train, Y_test = datasets.get_mvmc(te_percent=0.20)


cl = CloudNet(train=1)
cl.train_model(X_train, Y_train, bt_s=50, eps=2)

print("cloud on")
cl_test = CloudNet(train=0)
# exit()
print("node on")
# instantiate node and cloud network
nodes = []
for i in range(6):
    nodes.append(Node(i))

node_outputs = []
for n, i in zip(nodes, range(len(nodes))):
    node_outputs.append(n.calculate(X_test[str(i)]))


cl_in, cl_out = cl_test.get_in_out_tensor()

feed = {}
for input, node_out in zip(cl_in, node_outputs):
    feed[input] = node_out


# prediction_res = cl_test.model.pred(node_outputs)

# with tf.Session() as ses:
#     tf.global_variables_initializer().run()
    # cl_test.model.load("cloud")
    # cl_in, cl_out = cl_test.get_in_out_tensor()
    # prediction_res = ses.run(cl_out, feed_dict=feed)

# pred_res = []
# for res in prediction_res:
#     pred_res.append([np.argmax(res)])

# print(acc(pred_res, Y_train))


with tf.Session() as ses:
    tf.global_variables_initializer().run()
    cl_test.model.load("cloud")
    cl_in, cl_out = cl_test.get_in_out_tensor()
    prediction_res = ses.run(cl_out, feed_dict=feed)

pred_res = []
for res in prediction_res:
    pred_res.append([np.argmax(res)])

print(acc(pred_res, Y_test))



print(cl.eval_model(X_train, Y_train))

