import keras.models
import keras.optimizers
import keras.layers
import keras
from node import CnnModel, CloudNet
from scipy.stats import bernoulli
import tensorflow as tf
import numpy as np

class PolicyNetwork:
    def __init__(self):
        self.pnet = CnnModel(d_size=6)
        self.alpha = 0.8
        self.inp_shape = 16, 16, 42
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
    
    def train(self, input_data, epoch):
        pre = tf.placeholder(tf.float32, shape=(None, 1), name="predicted")#30 feature
        pnet_in, pnet_out = self.policy_network.get_in_out_tensor()

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
        print(loss)

        loss = tf.reduce_mean()
        ses = tf.InteractiveSession()
        ses.run(tf.global_variables_initializer())
        ses.close()


if __name__ == '__main__':
    #pn = PolicyNetwork()
    rl = RL()
    rl.train(None, None)