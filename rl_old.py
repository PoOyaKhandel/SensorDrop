import keras.models
import keras.optimizers
import keras.layers
import keras
from node import CnnModel, CloudNet
from scipy.stats import bernoulli
import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt


class PolicyNetwork:
    def __init__(self, train):
        self.pnet = CnnModel(d_size=6)
        self.train = train
        self.alpha = 0.8
        self.filter_num = 2
        self.inp_shape = 16, 16, self.filter_num
        self.input_tensor = self.pnet.add_inputs(inp_shape=self.inp_shape, num=6, name="pnet_input")
        convp_tensor = self.pnet.add_convp(inputs=self.input_tensor, parallel=-1, name="pnet_conv_1st")
        convp2_tensor = self.pnet.add_convp(inputs=convp_tensor, parallel=0, name="pnet_conv_2nd")
        #our model
        self.output_tensor = self.pnet.add_fully(convp2_tensor, flatten=1, name="pnet_fully")
        self.pnet.create_model(self.input_tensor, self.output_tensor, 0)
        # self.model = keras.models.Model(self.input_tensor, self.output_tensor)
        if self.train == 0:
            #self.pnet.create_model(self.input_tensor, self.output_tensor, 0)
            self.pnet.load("policy")

    def get_in_out_tensor(self):
        return self.input_tensor, self.output_tensor

    def feed(self, input_data):
        return self.pnet.pred(input_data)


class RL:
    def __init__(self):
        self.reward_minus_const = -2.0
        self.device_count = 6
        self.policy_network = PolicyNetwork(1)
        self.alpha=.9
        #self.cln = CloudNet(0)

    def reward(self, device_n, prediction):
        a = tf.multiply([(1 - 0*(device_n/6)**2)], tf.transpose(prediction))
        b = tf.multiply([tf.constant(self.reward_minus_const)], tf.transpose((1 - prediction)))
        return tf.add(a, b)


    def train(self, input_data, y_label, epoch):
        ses = tf.InteractiveSession()
        ses.run(tf.global_variables_initializer())
        
        batch_size = 25
        # zer = np.zeros_like(input_data[0][0])
        plot_list = []
        for e_itr in range(epoch):
            print("iteration Num:", e_itr)

            idx=np.random.permutation(input_data[0].shape[0])
            print(idx.shape)
            for iter in range(input_data[0].shape[0]//batch_size):
                current_batch=idx[int(iter*batch_size):int((iter+1)*batch_size)]
                x_cl= np.take(input_data,current_batch,axis=1)
                x_cl_original=x_cl.copy()
                y_label_batch=y_label[current_batch,:]

                f_dict = {}
                for pi, id in zip(pnet_in, x_cl):
                    f_dict[pi] = id
                u_decided = ses.run(u, feed_dict=f_dict)

                # print(u_decided)
                if iter<u_decided.shape[1]:
                   u_decided=np.ones(u_decided.shape)
                   for temp_l in range(iter):
                       u_decided[:,u_decided.shape[1]-1-temp_l]=0
                #print("==========")
                #print(u_decided)
                # exit()


                for n in range(x_cl[0].shape[0]):
                    avg_inp=np.zeros_like(input_data[0][0])
                    num_active=0
                    for i in range(u_decided.shape[1]):
                        if u_decided[n, i] == 1:
                            avg_inp += x_cl[i][n]
                            num_active += 1  
                    for i in range(u_decided.shape[1]):
                        if num_active>0:
                            if u_decided[n, i] == 0:
                                x_cl[i][n] = avg_inp/num_active
                        else:
                                x_cl[i][n] = 0*avg_inp
                        #else:
                                #x_cl[i][n] = 1.0*avg_inp
                    # print(num_active)

                input_dict = {}
                for pi, id in zip(cl_in, x_cl):
                    input_dict[pi] = id



                prediction_res = ses.run([cl_out], feed_dict=input_dict)
                # print("here1")
                # print(prediction_res)
                prediction_res = prediction_res[0]
                prediction_res = np.argmax(prediction_res, axis=1)
                # print("here2")
                # print(prediction_res)
                prediction_res = prediction_res.reshape(prediction_res.shape[0], 1)
                # print("here3")
                # print(prediction_res)
                a = prediction_res.copy()
                a[prediction_res == y_label_batch] = 1
                a[prediction_res != y_label_batch] = 0
                # print("here4")
                # print(a)
                # exit()

                input_dict = {pre: a}
                for pi, id in zip(pnet_in, x_cl):
                    input_dict[pi] = id
                _,loss_v, u_v,u_hat_v,pnet_out_v = ses.run([optimizer,loss, u,u_hat,pnet_out], feed_dict=input_dict)

                print(loss_v, end=",")

            print("")
            input_dict = {}
            for pi, id in zip(cl_in, input_data):
                input_dict[pi] = id                

            prediction_res = ses.run([cl_out], feed_dict=input_dict)
            prediction_res = prediction_res[0]
            prediction_res = np.argmax(prediction_res, axis=1)
            prediction_res = prediction_res.reshape(prediction_res.shape[0], 1)
            a = prediction_res.copy()
            a[prediction_res == y_label] = 1
            a[prediction_res != y_label] = 0

            input_dict = {pre: a}
            for pi, id in zip(pnet_in, input_data):
                input_dict[pi] = id


            policy_output,loss_list = ses.run([pnet_out, loss], feed_dict=input_dict)            


            print("-"*50)
            ss = np.argwhere(policy_output > 0.5)
            print(ss.shape[0], ss.shape)
            print("lost_list", loss_list)
            plot_list.append(np.round(policy_output))
            print("-"*50)


        print("loop_finished")
        self.policy_network.pnet.model.save_weights("policy_weights.h5")

        for p in plot_list[1:5]:
            f = plt.figure()
            for i in range(p.shape[1]): 
                ax = f.add_subplot(p.shape[1], 1, i+1) 
                ax.plot(p[:,i], label=str(i)) 
                ax.legend()
        for p in plot_list[-5:-1]:
            f = plt.figure()
            for i in range(p.shape[1]): 
                ax = f.add_subplot(p.shape[1], 1, i+1) 
                ax.plot(p[:,i], label=str(i)) 
                ax.legend()
        plt.show()




if __name__ == '__main__':
    rl = RL()
    rl.train(None, None)
