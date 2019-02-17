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
import math


class Enviroment_e:
    def __init__(self, env):
        self.reward_minus_const = 2.0
        self.device_count = 6
        self.cl_rl=env
        self.cloud_model=env.model_cloud
        
        # self.cl_in, self.cl_out = cl.get_in_out_tensor()
        
    def reward_calc(self, device_n, prediction):
        # print(device_n)
        a = np.multiply([10*(20 - 10.0*(device_n/6)**2)+self.reward_minus_const], np.transpose((prediction)))
        b = np.multiply([self.reward_minus_const], np.transpose((1 - prediction)))
        return np.add(a, b)

    def step(self, x_cl,y_label_batch, action, ses):

        prediction_res = self.cl_rl.calculate_claud(x_cl,action,apply_action=1)
        
        prediction_res_output=prediction_res.copy
        # print("here1")
        # print(prediction_res)
        prediction_res = prediction_res
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
    def __init__(self,env,ses,train=1,name="train"):
        self.device_count = 6
        self.train = train
        self.policy_filter_num = 20
        self.input_filter_num = 7
        self.rl_inp_shape = (None,16, 16, self.input_filter_num*self.device_count)
        self.kernel_size = (3, 3)
        self.pool_size = (2, 2)
        self.name=name

        self.Env=Enviroment_e(env)
        self.ses=ses
        if self.train==1:
            self.alpha=.1
        else:
            self.alpha=1

        self.train_loop_num=0


        # self.policy_network = PolicyNetwork(train=1)
        self.Make_RL()


    def Make_RL(self):

        with tf.variable_scope(self.name+"RL"): #All the convolution layers

            # pre = tf.placeholder(tf.float32, shape=(None, 1), name="predicted")#30 feature
            self.Advantage_input = tf.placeholder(tf.float32, shape=(None, 1), name="reward")#30 feature
            self.Iter_num = tf.placeholder(tf.int32, shape=(None, 1), name="reward")

            self.input_tensor = tf.placeholder(tf.float32, shape=self.rl_inp_shape, name="pnet_input")
            # self.output_placeholder = tf.placeholder(tf.float32, shape=(None, self.device_count), name="pnet_out")
            
            # Convolutional Layer #1 and Pooling Layer #1
            conv1 = tf.layers.conv2d(
                inputs=self.input_tensor,
                filters=self.policy_filter_num,
                kernel_size=self.kernel_size ,
                padding="same",
                activation=tf.nn.relu)        
            
            pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=self.pool_size,strides=1)

            # Convolutional Layer #2 and Pooling Layer #2
            conv2 = tf.layers.conv2d(
                inputs=pool1,
                filters=self.policy_filter_num,
                kernel_size=self.kernel_size ,
                padding="same",
                activation=tf.nn.relu)        
            
            pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=self.pool_size,strides=1)
            p2_shape=pool2.shape

            # Dense Layer
            pool2_flat = tf.reshape(pool2, [-1, p2_shape[1] * p2_shape[2] * p2_shape[3]])
            dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
            # dense = tf.layers.dense(inputs=dense, units=100, activation=tf.nn.relu)
            # dropout = tf.layers.dropout(
            #     inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)
            dropout = dense

            self.policy_out = tf.layers.dense(inputs=dropout, units=self.device_count,activation=tf.nn.sigmoid)
            # self.policy_out = tf.layers.dense(inputs=dropout, units=self.device_count,activation=tf.nn.softmax)


            self.pnet_in, self.pnet_out = self.input_tensor, self.policy_out


            # self.alpha= tf.cond(tf.equal(self.signal, 1), self.alpha/2,self.alpha)
        


            pnet_out_exp = self.pnet_out*self.alpha + (1-self.pnet_out)*(1-self.alpha)
            self.u = pnet_out_exp > tf.random_uniform(tf.shape(pnet_out_exp))
            self.u = tf.cast(self.u, tf.float32)


            # print(u_decided)
            # print(self.Iter_num.shape)
            # exit()
            # if tf.less(self.Iter_num , self.u.shape[1]):
            # if self.train_loop_num<self.device_count:
            #     self.u=tf.ones(self.u.shape)
            #     for temp_l in range(self.train_loop_num):
            #         self.u[:,self.u.shape[1]-1-temp_l]=0

            # def low_iter_val():
            #     tmp=tf.ones(shape=)
            #     for temp_l in range(self.Iter_num):
            #         tmp[:,self.u.shape[1]-1-temp_l]=0
            #     return tmp

            # def high_iter_val():
            #     return self.u

            # # print(self.u.shape[1])
            # self.u = tf.cond(tf.less(self.Iter_num, tf.constant(self.device_count)), low_iter_val(), high_iter_val())

  

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
            neg_log_prob=-1*(temp)

            # Vahid: not sure about the soft_max_cross_entropy!!
            # neg_log_prob = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.pnet_out, labels=self.u)

            # neg_log_prob = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.pnet_out, labels=self.u)
            # temp_old = tf.multiply(temp, Advantage)
            # loss_old = -1 * tf.reduce_mean(temp_old)

            neg_J = tf.multiply((neg_log_prob), tf.stop_gradient(self.Advantage_input))
            loss =  tf.reduce_mean(neg_J)

            s_cliped = tf.clip_by_value(self.pnet_out,   1e-15, 1-1e-15)
            entropy_loss = -tf.reduce_mean(tf.multiply(s_cliped,tf.log(s_cliped)))

            loss=(loss-entropy_loss)
            self.final_loss=(loss)
            # self.pnet.create_model(self.input_tensor, self.output_tensor, 0)
            # self.model = keras.models.Model(self.input_tensor, self.output_tensor)

        all_vars = tf.trainable_variables()
        rl_vars = [v for v in all_vars if v.name.startswith(self.name+"RL")]
        


        # print(rl_vars)
        # exit()
        # print(rl_vars_2)
        # print(map_vars)
        # exit()
        # map_vars={}
        # rl_vars_map = [map_vars{v.name.startswith(self.name+"RL")}=v.name.startswith("train_RL")
                            # for v in all_vars if v.name.startswith(self.name+"RL")]


        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.final_loss,var_list=rl_vars)


        uninitialized_vars = []
        for var in tf.global_variables():
            try:
                # print(var)
                self.ses.run(var)
            except tf.errors.FailedPreconditionError:
                # print("no - ", var)
                uninitialized_vars.append(var)


        init_new_vars_op = tf.variables_initializer(uninitialized_vars)
        self.ses.run(init_new_vars_op)


        if self.train == 0:
            rl_vars_2= [str(v).replace("test","train") for v in rl_vars]
            map_vars={}
            for a,b in zip(rl_vars,rl_vars_2):
                a_list= str(a).split('\'')
                b_list= str(b).split('\'')
                map_vars[a_list[1]]=b_list[1].replace('\'','')

            #self.pnet.create_model(self.input_tensor, self.output_tensor, 0)
            # self.saver.restore(self.ses, "./policy_weights.ckpt")
            # self.saver = tf.train.Saver(var_list=map_vars)
            self.saver = tf.train.Saver()
            self.saver.restore(self.ses, "./policy_weights.ckpt")
            print("Model restored.")

            # self.model.load_weights("policy_weights.h5", by_name=False)

    def calc_avg_cloud(self,x_cl, action=0,apply_action=1):

        # print(len(x_cl))

        # print(x_cl[0].shape)
        avg_batch=[]


        for n in range(x_cl[0].shape[0]):
            avg_inp=np.zeros_like(x_cl[0][0])
            if  (apply_action==0):            
                num_active=0
                for i in range(len(x_cl)):
                        avg_inp += x_cl[i][n]
                        num_active += 1  

            else:    
                num_active=0
                for i in range(len(x_cl)):
                    if (int(action[n, i]) == 1):
                        avg_inp += x_cl[i][n]
                        # avg_inp += x_cl[i][n]
                        num_active += 1

            
            if num_active==0:
                avg_inp=0*avg_inp
            else:
                avg_inp=avg_inp/num_active  

            avg_batch.append(avg_inp)
            
        # print("-------")
        # print(avg_inp.shape)
        # print(np.array(avg_batch).shape)
        # exit()
        return avg_batch

    def calc_inp_rl(self,x_cl):

        # print(len(x_cl))
        rl_input=x_cl[0]


        for tmp in range(self.device_count-1):
            rl_input=np.concatenate((rl_input,x_cl[tmp+1]),axis=3)



        # print(x_cl[0].shape)
        return rl_input


    def train_RL(self, input_data, y_label, epoch):
        # self.ses.run(tf.global_variables_initializer())
        # self.ses = tf.InteractiveSession()
        # .run(tf.global_variables_initializer())

        batch_size = 6
        # zer = np.zeros_like(input_data[0][0])
        plot_list = []
        for e_itr in range(epoch):
            print("iteration Num:", e_itr)

            # print(e_itr)
            if e_itr<5:
                self.alpha=.1

            if e_itr==10:
                self.alpha=.7

            
            if e_itr%5==1:
                self.alpha=np.sqrt(self.alpha)
            print(self.alpha)

            idx=np.random.permutation(input_data[0].shape[0])
            print(idx.shape)

            train_loop_num=0
            for iter in range(input_data[0].shape[0]//batch_size):
                train_loop_num=train_loop_num+1
                current_batch=idx[int(iter*batch_size):int((iter+1)*batch_size)]
                x_cl= np.take(input_data,current_batch,axis=1)
                x_cl_original=x_cl.copy()

                # cloud_avg_inp=self.calc_avg_cloud(x_cl, action=0, apply_action=0)
                rl_input= self.calc_inp_rl(x_cl)
                # print("==++++===")
                # print((rl_input.shape))
                # exit()



                y_label_batch=y_label[current_batch,:]

                f_dict = {}
                # for pi, id in zip(self.pnet_in, x_cl):
                #     # print(pi)
                #     f_dict[pi] = id
                f_dict[self.input_tensor]=rl_input
                f_dict[self.Iter_num]=np.full(shape=(batch_size,1),fill_value=e_itr)

                    
                u_decided,u_hat_decided = self.ses.run([self.u,self.u_hat], feed_dict=f_dict)
                
                Reward_value_u,_,_= self.Env.step(x_cl,y_label_batch,u_decided,self.ses)
                Reward_value_u_hat,_,_= self.Env.step(x_cl,y_label_batch,u_hat_decided,self.ses)

                Advantage_value=Reward_value_u
                # Advantage_value=Reward_value_u-Reward_value_u_hat
                

                input_dict = {}
                # for pi, id in zip(self.pnet_in, x_cl):
                #     input_dict[pi] = id
                input_dict[self.input_tensor]=rl_input
                input_dict[self.Iter_num]=np.full(shape=(batch_size,1),fill_value=e_itr)
                input_dict[self.Advantage_input]=Advantage_value.T


                _,loss_v = self.ses.run([self.optimizer,self.final_loss], feed_dict=input_dict)

                # print(loss_v,"(",Reward_value_u,",",Reward_value_u_hat,")")
                # print(loss_v,"(",np.sum(u_decided,axis=1),",",np.sum(u_hat_decided,axis=1) ,")")
                if math.isnan(loss_v):
                    print(Advantage_value)
                    print(u_decided)
                    print(u_hat_decided)
                    exit()
                
                print(loss_v,"(",np.mean(Reward_value_u),") ", end=",")
                # print(loss_v, end=",")
                

            print("")
            rl_input= self.calc_inp_rl(input_data)
            # avg_inp=self.calc_avg_cloud(input_data, action=0, apply_action=0)
            f_dict = {}
            # for pi, id in zip(self.pnet_in, input_data):
            #     f_dict[pi] = id
            f_dict[self.input_tensor]=rl_input
            f_dict[self.Iter_num]=np.full(shape=(input_data[0].shape[0],1),fill_value=5000)
            u_decided,u_hat_decided = self.ses.run([self.u,self.u_hat], feed_dict=f_dict)
            
            Reward_value_u,_,_= self.Env.step(input_data,y_label,u_decided,self.ses)

            input_dict = {}
            # for pi, id in zip(self.pnet_in, input_data):
            #     input_dict[pi] = id
            input_dict[self.input_tensor]=rl_input
            input_dict[self.Iter_num]=np.full(shape=(input_data[0].shape[0],1),fill_value=5000)
            input_dict[self.Advantage_input]=Reward_value_u.T

            loss_list = self.ses.run([self.final_loss], feed_dict=input_dict)

            print("-"*50)
            ss = np.argwhere(u_decided > 0.5)
            print(ss.shape[0], ss.shape)
            print("mean_advantage", np.mean(Reward_value_u))
            print("lost_list", loss_list)
            plot_list.append(np.round(u_decided))
            print("-"*50)


        print("loop_finished")
        # self.policy_network.model.save_weights("policy_weights.h5")
        self.saver = tf.train.Saver()
        save_path = self.saver.save(self.ses, "./policy_weights.ckpt")
        print("Model saved in path: %s" % save_path)

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
