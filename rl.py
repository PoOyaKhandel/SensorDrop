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
            self.pnet.create_model(self.input_tensor, self.output_tensor, 0)
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
        a = tf.multiply([(1 - (device_n/6)**2)], tf.transpose(prediction))
        b = tf.multiply([tf.constant(self.reward_minus_const)], tf.transpose((1 - prediction)))
        return tf.add(a, b)

    def train(self, input_data, y_label, epoch):
        pre = tf.placeholder(tf.float32, shape=(None, 1), name="predicted")#30 feature
        pnet_in, pnet_out = self.policy_network.get_in_out_tensor()

        cl = CloudNet(0)
        cl_in, cl_out = cl.get_in_out_tensor()

        u = pnet_out < tf.random_uniform(tf.shape(pnet_out))
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
        temp = tf.multiply(temp, reward)
        loss = tf.reduce_mean(temp)
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

        ses = tf.InteractiveSession()
        ses.run(tf.global_variables_initializer())
        f_dict = {}
        for pi, id in zip(pnet_in, input_data):
            f_dict[pi] = id
        policy_output = ses.run(pnet_out, feed_dict=f_dict)
        batch_size = 100
        zer = np.zeros_like(input_data[0][0])
        plot_list = []
        for e_itr in range(epoch):
            x_cl = input_data.copy()
            # print(len(x_cl))
            # print(x_cl[0].shape)
            # exit()
            # print(zer.shape)
            # exit()
            for i in range(policy_output.shape[1]):
                for n in range(x_cl[0].shape[0]):
                    if policy_output[n, i] < 0.5:
                        x_cl[i][n] = zer

            input_dict = {}
            for pi, id in zip(cl_in, x_cl):
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
            _, policy_output = ses.run([optimizer, pnet_out], feed_dict=input_dict)
            # print(np.count_nonzero(policy_output, axis=0))
            ss = np.argwhere(policy_output > 0.5)
            print(ss.shape[0])
            plot_list.append(np.round(policy_output))
            
        for p in plot_list:
            f = plt.figure()
            for i in range(p.shape[1]): 
                ax = f.add_subplot(p.shape[1], 1, i+1) 
                ax.plot(p[:,i], label=str(i)) 
                ax.legend()
        plt.show()

        print("loop_finished")
        self.policy_network.pnet.model.save_weights("policy_weights.h5")


if __name__ == '__main__':
    rl = RL()
    rl.train(None, None)
