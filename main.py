from dataset import datasets
from node import  CloudNet
from rl import RL
import numpy as np
from sklearn.metrics import accuracy_score as acc
import tensorflow as tf
from keras.utils.vis_utils import plot_model
# import os
from keras.utils import to_categorical

# # os.environ["PATH"] += os.pathsep + 'C:\Program Files (x86)\Graphviz2.38\bin\'

# print(os.environ["PATH"])
ses = tf.InteractiveSession()
# ses.run(tf.global_variables_initializer())
# np.set_printoptions(precision=2)

X_train, X_test, Y_train, Y_test = datasets.get_mvmc(te_percent=0.20)
# X_train, X_test, Y_train, Y_test = datasets.get_mvmc_concat(te_percent=0.20)

print(X_train['0'].shape)
print(X_test['0'].shape)


# exit()
#  train the model

iftrain_CloudNet=0
iftrain_RLNet =1
iftest_compl = 1

if iftrain_CloudNet==1:
    # tf.reset_default_graph()    
    cl_train = CloudNet(train=1)
    cl_train.train_model(X_train, Y_train, bt_s=50, eps=30)

# # instantiate node and cloud network
# node = []
# for i in range(6):
#     node.append(Node(i))
# # exit()

# cl_t = CloudNet(train=1)
# plot_model(cl_t.model_cloud, to_file='cloud.png')
# plot_model(cl_t.node[1], to_file='node.png')
# plot_model(cl_t.model, to_file='all.png')

if iftrain_RLNet==1:
    # tf.reset_default_graph()
    cl_rl = CloudNet(train=0)

    node_output = []    
    for l in range(6):
        print("input node", l, "is processing.")
        node_output.append(cl_rl.node[l].predict(X_train[str(l)].reshape((-1, 32, 32, 3))))

    # rl = RL(cl_rl, ses=ses, train=1,  name='train_')
    rl = RL(cl_rl, ses=ses, train=1)
    rl.train_RL(node_output, Y_train, 25)

# exit()

# node_output = []
# for l in range(6):
#     node_output.append(node[l].calculate(X_test[str(l)]))
if iftest_compl==1:
    # tf.reset_default_graph()    

    cl_t = CloudNet(train=0)

    cl_t.model_all.summary()
    cl_t.model_cloud.summary()
    cl_t.node[0].summary()



    # X_test=X_train
    # Y_test=Y_train

    print("here1")
    # print(cl.eval_cloud_model(node_output, Y_test))
    x = []
    for l in range(6):
        x.append(X_test[str(l)].reshape((-1, 32, 32, 3)))
    print(cl_t.eval_comp_model(x, Y_test))
    print("here2")


    node_output_t = []
    for l in range(6):
        print("Test --input node", l, "is processing.")
        node_output_t.append(cl_t.node[l].predict(X_test[str(l)].reshape((-1, 32, 32, 3))))

    # print(cl_t.model_cloud.evaluate(node_output, y_cat))

    print(cl_t.calculate_claud(node_output_t,action=0,apply_action=0))
    # print(cl_t.model_cloud.predict(node_output))
    # print(cl_t.eval_cloud_model(node_output, Y_test))
    print("here2.5")
    print(cl_t.evaluate_claud(node_output_t,Y_test,action=0,apply_action=0))



    print("here3")
    if iftrain_RLNet==1:
        rl_t=   rl 
    else:
        # rl_t = RL(cl_t, ses=ses, train=0, name='test_')
        rl_t = RL(cl_t, ses=ses, train=0)

    # policy_for_test = policy_net.model.predict(node_output_t)

    # pnet_in, pnet_out = policy_net.get_in_out_tensor()
    print(rl_t.input_tensor)
    print(len(node_output_t))
    print((node_output_t[0].shape))
    
    
    rl_inp=rl_t.calc_inp_rl(node_output_t)
    f_dict = {}
    # for pi, id in zip(rl_t.input_tensor, node_output_t):
    #     print(pi)
    #     f_dict[pi] = id
    f_dict[rl_t.input_tensor]=rl_inp
    f_dict[rl_t.Iter_num]=np.full(shape=(node_output_t[0].shape[0],1),fill_value=5000)

    pnet_out = ses.run([rl_t.policy_out], feed_dict=f_dict)

    pnet_out=pnet_out[0]
    print(len(pnet_out))
    print(pnet_out[0].shape)
    print((pnet_out))
    policy_for_test = pnet_out > 0.5*np.ones(shape=pnet_out.shape)
    # policy_for_test = float(policy_for_test)

    print(policy_for_test.shape)
    print(np.count_nonzero(policy_for_test, axis=0))
    print("here4")

    print("------RL Selected-----")
    y = cl_t.calculate_claud(node_output_t, policy_for_test,apply_action=1)
    y = np.argmax(y, axis=1)

    print(y.shape)
    print(acc(y, Y_test))

    print("------all in-----")
    policy_for_test=np.ones(policy_for_test.shape)
    y = cl_t.calculate_claud(node_output_t, policy_for_test,apply_action=1)
    y = np.argmax(y, axis=1)

    print(y.shape)
    print(acc(y, Y_test))

