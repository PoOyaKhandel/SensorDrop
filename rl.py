import keras.models
import keras.optimizers
import keras.layers
import keras
from node import CnnModel, CloudNet
from scipy.stats import bernoulli
import tensorflow as tf
import numpy as np
import time

class PolicyNetwork:
    def __init__(self):
        self.pnet = CnnModel(d_size=4)
        self.alpha = 0.8
        self.inp_shape = 16, 16, 7
        self.input_tensor = self.pnet.add_inputs(inp_shape=self.inp_shape, num=6, name="pnet_input")
        convp_tensor = self.pnet.add_convp(inputs=self.input_tensor, parallel=-1, name="pnet_conv_1st")
        convp2_tensor = self.pnet.add_convp(inputs=convp_tensor, parallel=0, name="pnet_conv_2nd")
        #our model
        self.output_tensor = self.pnet.add_fully(convp2_tensor, flatten=1, name="pnet_fully")
        self.pnet.create_model(self.input_tensor, self.output_tensor, 0)

    def get_in_out_tensor(self):
        return self.input_tensor, self.output_tensor

    def feed(self, input):
        return self.pnet.predict(input)    
        


class RL:
    def __init__(self):
        self.reward_minus_const = -0.1
        self.device_count = 6
        self.policy_network = PolicyNetwork()
        #self.cln = CloudNet(0)


    def reward(self, device_n, prediction):
        a = tf.multiply([(1 - (device_n/6)**2)], tf.transpose(prediction))
        b = tf.multiply([tf.constant(self.reward_minus_const)], tf.transpose((1 - prediction)))
        return  tf.add(a, b)


    def get_loss(self):
        pass

    def bernoulli(self, s):
        pass
    
    def train(self, input_data, y_label, epoch):
        pre = tf.placeholder(tf.float32, shape=(None, 1), name="predicted")#30 feature
        pnet_in, pnet_out = self.policy_network.get_in_out_tensor()

        cl = CloudNet(0)
        cl_in, cl_out = cl.get_in_out_tensor()

        # print(cl_in)
        # exit()

        # print(type(pre), type(pnet_in[0]))
        # exit()

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
            temp = tf.add(temp , tf.math.log(a))

        # temp = tf.reduce_sum(temp, axis=1)
        reward = self.reward(tf.count_nonzero(u, axis=1, dtype=tf.float32), pre)
        temp = tf.multiply(temp, reward)
        loss = tf.reduce_mean(temp)
        # print(loss)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)

        ses = tf.InteractiveSession()
        # with tf.Session() as ses:
        ses.run(tf.global_variables_initializer())
        f_dict = {}
        for pi, id in zip(pnet_in, input_data):
            f_dict[pi] = id
        policy_output = ses.run(pnet_out, feed_dict=f_dict)
        # ses.close()
        batch_size = 100
        zer = np.zeros_like(input_data[0])
        for e_itr in range(epoch):
            train_loss = 0
            # start = time.time()
            # for indx  in range(input_data[0].shape[0]//batch_size):

            x_cl = input_data.copy()
            for i in range(policy_output.shape[1]):
                for n in range(x_cl[0].shape[0]):
                    if policy_output[n, i] == 0:
                        x_cl[i, n] = zer

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

        print("loop_finished")



if __name__ == '__main__':
    rl = RL()
    rl.train(None, None)