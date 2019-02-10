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
        self.inp_shape = 16, 16, 7
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
        self.reward_minus_const = -0.1
        self.device_count = 6
        self.policy_network = PolicyNetwork(1)
        #self.cln = CloudNet(0)

    def reward(self, device_n, prediction):
        a = tf.multiply([(1 - 0.3*(device_n/6)**2)], tf.transpose(prediction))
        b = tf.multiply([tf.constant(self.reward_minus_const)], tf.transpose((1 - prediction)))
        return tf.add(a, b)

    def train(self, input_data, y_label, epoch):
        pre = tf.placeholder(tf.float32, shape=(None, 1), name="predicted")#30 feature
        pnet_in, pnet_out = self.policy_network.get_in_out_tensor()

        cl = CloudNet(0)
        cl_in, cl_out = cl.get_in_out_tensor()

        u = pnet_out > tf.random_uniform(tf.shape(pnet_out))
        u = tf.cast(u, tf.float32)
        temp = tf.constant(0.0)
        print(pnet_out.shape)
        for i in range(pnet_out.shape[1]):
            s_t = tf.transpose([pnet_out[ :, i]])
            s = tf.transpose(s_t)
            u_t = tf.transpose([u[:, i]])
            u_ = tf.transpose(u_t)
            m_su = tf.multiply(s, u_)
            m_1_s_1_u = tf.multiply(1-s, 1-u_)
            a = tf.add(m_su, m_1_s_1_u)
            temp = tf.add(temp, tf.math.log(a))

        reward = self.reward(tf.count_nonzero(u, axis=1, dtype=tf.float32), pre)

        u_hat = pnet_out > 0.5*tf.ones(shape=tf.shape(pnet_out))
        u_hat = tf.cast(u_hat, tf.float32)
        reward_hat = self.reward(tf.count_nonzero(u_hat, axis=1, dtype=tf.float32), pre)

        Advantage=reward#-reward_hat

        temp = tf.multiply(temp, Advantage)
        loss = tf.reduce_mean(temp)
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(-loss)

        ses = tf.InteractiveSession()
        ses.run(tf.global_variables_initializer())
        f_dict = {}
        # for pi, id in zip(pnet_in, input_data):
        #     f_dict[pi] = id
        # policy_output = ses.run(pnet_out, feed_dict=f_dict)
        
        
        batch_size = 15
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

                # print(u_decided.shape)
                # print(u_decided[1:5,1])
                # print("------")
                # print(x_cl.shape)
                # print(x_cl[1,1:5,1:6,1,1])
                # print("------")
                # print(x_cl_original[1,1:5,1:6,1,1])
                # print("------")
                # print(avg_inp.shape)
                # print(avg_inp[1:6,1,1])
                # exit()
                # for i in range(policy_output.shape[1]):
                #     for n in range(x_cl[0].shape[0]):
                #         if policy_output[n, i] < 0.5:
                #             x_cl[i][n] = zer

                input_dict = {}
                for pi, id in zip(cl_in, x_cl):
                    input_dict[pi] = id

                prediction_res = ses.run([cl_out], feed_dict=input_dict)
                prediction_res = prediction_res[0]
                prediction_res = np.argmax(prediction_res, axis=1)
                prediction_res = prediction_res.reshape(prediction_res.shape[0], 1)
                a = prediction_res.copy()
                a[prediction_res == y_label_batch] = 1
                a[prediction_res != y_label_batch] = 0
                # print(a.shape)
                # print(y_label_batch.shape)
                # print(prediction_res.shape)
                # exit()

                input_dict = {pre: a}
                for pi, id in zip(pnet_in, x_cl):
                    input_dict[pi] = id
                _,loss_v, u_v,u_hat_v,pnet_out_v = ses.run([optimizer,loss, u,u_hat,pnet_out], feed_dict=input_dict)
                # print(np.count_nonzero(policy_output, axis=0))
                # ss = np.argwhere(policy_output > 0.5)
                # print(ss.shape[0])
                # plot_list.append(np.round(policy_output))
                # print(pnet_out_v)
                # print(u_v)
                # print(u_hat_v)
                # exit()
                print(loss_v,end=",")

            print("")
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
            # print(input_dict.keys())
            # exit()

            policy_output,loss_list = ses.run([pnet_out, loss], feed_dict=input_dict)            
            # print(np.count_nonzero(policy_output, axis=0))
            # print(policy_output)
            # print(policy_output.shape)

            print("----------")
            ss = np.argwhere(policy_output > 0.5)
            print(ss.shape[0])
            print((loss_list))
            plot_list.append(np.round(policy_output))
            print("----------")


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
