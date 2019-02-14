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
        self.alpha = 0.9
        self.filter_num = 7
        self.inp_shape = 16, 16, self.filter_num

        self.input_tensor = self.pnet.add_inputs(inp_shape=self.inp_shape, num=6, name="pnet_input")
        convp_tensor = self.pnet.add_convp_notbinary(inputs=self.input_tensor, parallel=-1, name="pnet_conv_1st")
        convp2_tensor = self.pnet.add_convp_notbinary(inputs=convp_tensor, parallel=0, name="pnet_conv_2nd")
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

class Enviroment_e:
    def __init__(self, env):
        self.reward_minus_const = -2.0
        self.device_count = 6
        self.cloud_in=env.cloud_input_tensor
        self.cloud_out=env.output_tensor
        
        # self.cl_in, self.cl_out = cl.get_in_out_tensor()
        
    def reward_calc(self, device_n, prediction):
        a = np.multiply([(1 - 1*(device_n/6)**2)], np.transpose(prediction))
        b = np.multiply([self.reward_minus_const], np.transpose((1 - prediction)))
        return np.add(a, b)

    def step(self, x_cl,y_label_batch, action, ses):

        for n in range(x_cl[0].shape[0]):
            avg_inp=np.zeros_like(x_cl[0][0])
            num_active=0
            for i in range(action.shape[1]):
                if int(action[n, i]) == 1:
                    avg_inp += x_cl[i][n]
                    num_active += 1  
            for i in range(action.shape[1]):
                if num_active>0:
                    if int(action[n, i]) == 0:
                        x_cl[i][n] = avg_inp/num_active
                else:
                        x_cl[i][n] = 0*avg_inp
                #else:
                        #x_cl[i][n] = 1.0*avg_inp
            # print(num_active)

        input_dict = {}
        for pi, id in zip(self.cloud_in, x_cl):
            # print(pi)
            # print(id.shape)
            input_dict[pi] = id
        # exit()
        prediction_res = ses.run(self.cloud_out, feed_dict=input_dict)
        prediction_res_output=prediction_res.copy
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

        self.reward = self.reward_calc(np.count_nonzero(action.astype(int), axis=1), a)

        return self.reward, prediction_res_output, a


class RL:
    def __init__(self,env,train=1):
        self.reward_minus_const = -2.0
        self.device_count = 6
        self.alpha=.9

        self.Env=Enviroment_e(env)

        self.policy_network = PolicyNetwork(train=1)
        self.Make_RL()

    def Make_RL(self):

        # pre = tf.placeholder(tf.float32, shape=(None, 1), name="predicted")#30 feature
        self.Advantage_input = tf.placeholder(tf.float32, shape=(None, 1), name="reward")#30 feature
        self.Iter_num = tf.placeholder(tf.int32, shape=(None, 1), name="reward")

        self.pnet_in, self.pnet_out = self.policy_network.get_in_out_tensor()

        pnet_out_exp = self.pnet_out*self.alpha + (1-self.pnet_out)*(1-self.alpha)
        self.u = pnet_out_exp > tf.random_uniform(tf.shape(pnet_out_exp))
        self.u = tf.cast(self.u, tf.float32)

        # # print(u_decided)
        # if self.Iter_num <self.u.shape[1]:
        #     self.u=tf.ones(self.u.shape)
        #     for temp_l in range(self.Iter_num):
        #         self.u[:,self.u.shape[1]-1-temp_l]=0
        # #print("==========")
        # #print(u_decided)
        # # exit()

        self.u_hat = self.pnet_out > 0.5*tf.ones(shape=tf.shape(self.pnet_out))
        self.u_hat = tf.cast(self.u_hat, tf.float32)


        temp = tf.constant(0.0)
        print(self.pnet_out.shape)
        for i in range(self.pnet_out.shape[1]):
            s_t = tf.transpose([self.pnet_out[ :, i]])
            s = tf.transpose(s_t)
            u_t = tf.transpose([self.u[:, i]])
            u_ = tf.transpose(u_t)
            m_su = tf.multiply(s, u_)
            m_1_s_1_u = tf.multiply(1-s, 1-u_)
            a = tf.add(m_su, m_1_s_1_u)
            temp = tf.add(temp, tf.math.log(a))

        # Vahid: not sure about the sign!!
        neg_log_prob=-1*(-1*temp)

        # Vahid: not sure about the soft_max_cross_entropy!!
        # neg_log_prob = tf.nn.softmax_cross_entropy_with_logits_v2(logits=pnet_out, labels=u)
        # temp_old = tf.multiply(temp, Advantage)
        # loss_old = -1 * tf.reduce_mean(temp_old)

        neg_J = tf.multiply(neg_log_prob, tf.stop_gradient(self.Advantage_input))
        loss =  tf.reduce_mean(neg_J)

        s_cliped = tf.clip_by_value(self.pnet_out,   1e-15, 1-1e-15)
        entropy_loss = -tf.reduce_mean(tf.multiply(s_cliped,tf.log(s_cliped)))

        # loss=(loss-entropy_loss)
        self.final_loss=(loss)

        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.final_loss)

    def train(self, input_data, y_label, epoch, ses):
        # ses.run(tf.global_variables_initializer())
        # ses = tf.InteractiveSession()
        # ses.run(tf.global_variables_initializer())
        uninitialized_vars = []
        for var in tf.global_variables():
            try:
                ses.run(var)
            except tf.errors.FailedPreconditionError:
                uninitialized_vars.append(var)

        init_new_vars_op = tf.variables_initializer(uninitialized_vars)
        ses.run(init_new_vars_op)

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
                for pi, id in zip(self.pnet_in, x_cl):
                    # print(pi)
                    f_dict[pi] = id

                f_dict[self.Iter_num]=np.full(shape=(batch_size,1),fill_value=e_itr)

                    
                u_decided,u_hat_decided = ses.run([self.u,self.u_hat], feed_dict=f_dict)
                
                Reward_value_u,_,_= self.Env.step(x_cl,y_label_batch,u_decided,ses)
                Reward_value_u_hat,_,_= self.Env.step(x_cl,y_label_batch,u_hat_decided,ses)

                Advantage_value=Reward_value_u
                # Advantage_value=Reward_value_u-Reward_value_u_hat
                
                input_dict = {}
                for pi, id in zip(self.pnet_in, x_cl):
                    input_dict[pi] = id
                input_dict[self.Iter_num]=np.full(shape=(batch_size,1),fill_value=e_itr)
                input_dict[self.Advantage_input]=Advantage_value.T

                _,loss_v = ses.run([self.optimizer,self.final_loss], feed_dict=input_dict)

                print(loss_v, end=",")

            print("")
            f_dict = {}
            for pi, id in zip(self.pnet_in, input_data):
                f_dict[pi] = id
            f_dict[self.Iter_num]=np.full(shape=(input_data[0].shape[0],1),fill_value=5000)
            u_decided,u_hat_decided = ses.run([self.u,self.u_hat], feed_dict=f_dict)
            
            Reward_value_u,_,_= self.Env.step(input_data,y_label,u_decided,ses)

            input_dict = {}
            for pi, id in zip(self.pnet_in, input_data):
                input_dict[pi] = id
            input_dict[self.Iter_num]=np.full(shape=(input_data[0].shape[0],1),fill_value=5000)
            input_dict[self.Advantage_input]=Reward_value_u.T

            loss_list = ses.run([self.final_loss], feed_dict=input_dict)

            print("-"*50)
            ss = np.argwhere(u_decided > 0.5)
            print(ss.shape[0], ss.shape)
            print("lost_list", loss_list)
            plot_list.append(np.round(u_decided))
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




# if __name__ == '__main__':
    # rl = RL()
    # rl.train(None, None)
