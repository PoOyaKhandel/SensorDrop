import keras.models
import keras.optimizers
import keras.layers
import keras
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from keras.utils.vis_utils import plot_model
import numpy as np
import tensorflow as tf


class CnnModel:
    filter_num = 7
    weightPath = "w.h5"

    def __init__(self, d_size,model_name='model'):
        self.model_ = None
        self.optimizer = None
        # self.loss = 'mean_squared_error'
        self.loss = 'categorical_crossentropy'

        @tf.custom_gradient
        def custom_activation(x):
            def grad(dy):
                return dy * tf.exp(x)/tf.pow(1 + tf.exp(x), 2)
            return tf.keras.backend.round(tf.keras.backend.sigmoid(x)), grad

        self.activation = custom_activation
        self.kernel_size = (3, 3)
        self.filter_num = CnnModel.filter_num
        self.input_shape = (32, 32, 3)
        self.pool_size = (2, 2)
        self.dense_len = d_size  # person, bus, car, not-present

    def __define_convp_notbinary(self, convp_in, name):
        """
        :param name: name of convp basic blocks
        :param convp_in: list of input layer for convp blocks
        :return: Convp Block output
        """
        conv2d_base = self.__define_conv2d_notbinary(name=name+"_conv2d")
        pooling_base = self.__define_max_pool(name=name+"_pooling")
        batch_norm_base = self.__define_batch_normalization(name=name+"_batch")
        # batch_norm_base = pooling_base
        output = []
        
        for n in range(len(convp_in)):
            # output.append(batch_norm_base(pooling_base(conv2d_base(convp_in[n]))))
            output.append((pooling_base(conv2d_base(convp_in[n]))))
        return output

    def __define_convp(self, convp_in, name):
        """
        :param name: name of convp basic blocks
        :param convp_in: list of input layer for convp blocks
        :return: Convp Block output
        """
        conv2d_base = self.__define_conv2d(name=name+"_conv2d")
        pooling_base = self.__define_max_pool(name=name+"_pooling")
        batch_norm_base = self.__define_batch_normalization(name=name+"_batch")
        output = []

        if (type(convp_in)==list):     
            for n in range(len(convp_in)):
                # output.append(batch_norm_base(pooling_base(conv2d_base(convp_in[n]))))
                output.append((pooling_base(conv2d_base(convp_in[n]))))
            return output
        else:
            return pooling_base(conv2d_base(convp_in))




    def __define_conv2d_notbinary(self, name):
        """
        :param name: block name
        :return: conv2d Layer
        """
        return keras.layers.Conv2D(self.filter_num, self.kernel_size, strides=(1, 1), activation='relu',
                                   padding='same',
                                   kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=1),
                                   name=name)


    def __define_conv2d(self, name):
        """
        :param name: block name
        :return: conv2d Layer
        """
        return keras.layers.Conv2D(self.filter_num, self.kernel_size, strides=(1, 1), activation=self.activation,
                                   padding='same',
                                   kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=1),
                                   name=name)

    def __define_max_pool(self, name):
        """
        :param name: block name
        :return: pooling Layer
        """
        return keras.layers.MaxPooling2D(pool_size=self.pool_size, padding='same', name=name)

    def __define_batch_normalization(self, name):
        """
        :param name: block name
        :return: Batch Layer
        """
        return keras.layers.BatchNormalization(name=name)

    def __define_flatten(self, name):
        """
        :param name: block name
        :return: Flatten Layer
        """
        return keras.layers.Flatten(name=name)

    def __define_fully(self, name):
        """
        :param name: block name
        :return: Fully Layer
        """
        #return keras.layers.Dense(self.dense_len, name=name, activation="sigmoid")
        return keras.layers.Dense(self.dense_len, name=name, activation="softmax")

    def __config_optimizer(self, lr=0.001, beta_1=0.9, beta_2=0.999):
        """
        :param lr: learning rate
        :param beta_1:
        :param beta_2:
        :return: None
        """
        self.optimizer = keras.optimizers.Adam(lr, beta_1, beta_2, epsilon=1e-8)

    def __compile_func_model(self):
        """
        Compile the model before training
        :return: None
        """
        self.__config_optimizer()
        self.model_.compile(loss=self.loss, optimizer=self.optimizer, metrics=['accuracy'])

    def create_model(self, inp, out, comp):
        """
        defining model with Functional API keras
        :param comp: if 1 compile the model
        :param inp: input layer of model
        :param out: output layer of model
        :return: None
        """
        self.model_ = keras.models.Model(inputs=inp, outputs=out)
        if comp == 1:
            self.__compile_func_model()

    def train_model(self, X, Y, btch_size, ep):
        """
        train the model
        :param X: train input
        :param Y: train output
        :param btch_size: batch size
        :param ep: #epochs
        :return: None
        """
        history = self.model_.fit(x=X, y=Y, batch_size=btch_size, epochs=ep, verbose=2)
        # summarize history for accuracy
        plt.plot(history.history['acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train'], loc='upper left')
        plt.show()
        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train'], loc='upper left')
        plt.show()
        json = self.model_.to_json()
        # print(json)
        # print(self.model.get_weights()[1].shape, self.model.get_weights()[2].shape)
        # print(self.model.get_weights)
        self.model_.save_weights("cloud_weights.h5")

    def eval_model(self, X, Y):
        """
        Evaluate trained model
        :param X: test input
        :param Y: test output
        :return: [loss, accuracy]
        """
        return self.model_.evaluate(X, Y)

    def add_inputs(self, inp_shape, num=1, name="input"):
        """
        :param name: input name
        :param num: number of inputs
        :param inp_shape: shape of each input
        :return: inputs Tensor
        """
        inputs = []
        if num != 0:
            for n in range(num):
                inputs.append(keras.layers.Input(shape=inp_shape, name=name+str(n+1)))
        else:
            raise NotImplementedError("num can not be zero")
        return inputs

    def add_convp_notbinary(self, inputs, parallel=0, name="?"):
        """
        :param name: convp blocks name
        :param inputs: gets inputs of set(s) of Convp blocks
        :param parallel: 1-> parallel convp + concatenate. 0-> single convp. -1-> concatenate+single convp
        :return: concatenated block of parallel convps
        """
        if parallel == 1:
            conv_out = self.__define_convp_notbinary(inputs, name)
            return keras.layers.average(conv_out, name=name+"concat")
        elif parallel == 0:
            conv_out = self.__define_convp_notbinary(inputs, name)
            return conv_out[0]
        elif parallel == -1:
            concat = keras.layers.average(inputs, name=name+"concat")
            print(concat)
            return self.__define_convp_notbinary([concat], name=name)
        else:
            raise NotImplementedError("wrong parallel input value")



    def add_convp(self, inputs, parallel=0, name="?"):
        """
        :param name: convp blocks name
        :param inputs: gets inputs of set(s) of Convp blocks
        :param parallel: 1-> parallel convp + concatenate. 0-> single convp. -1-> concatenate+single convp
        :return: concatenated block of parallel convps
        """
        if parallel == 1:
            conv_out = self.__define_convp(inputs, name)
            return keras.layers.average(conv_out, name=name+"concat")
        elif parallel == 0:
            conv_out = self.__define_convp(inputs, name)
            return conv_out
        elif parallel == -1:
            concat = keras.layers.average(inputs, name=name+"concat")
            print(concat)
            return self.__define_convp([concat], name=name)
        else:
            raise NotImplementedError("wrong parallel input value")

    def add_fully(self, fully_in, flatten=1, name="?"):
        """
        :param name: block name
        :param fully_in: fully block input tensor
        :param flatten: if 1 also add flatten layer before fully layer
        :return:
        """
        if flatten == 1:
            flat = self.__define_flatten(name=name+"flatten")(fully_in)
            return self.__define_fully(name=name+"fully")(flat)
        else:
            return self.__define_fully(name=name+"fully")(fully_in)

    def load(self, weights):
        """
        loading model weights
        :return: None
        """
        # self.model_.load_weights(weights+"_weights.h5", by_name=True)
        self.model_.load_weights(weights+"_weights.h5", by_name=False)

    def pred(self, x):
        """
        :param x: input vector
        :return: model prediction of x
        """
        return self.model_.predict(x)

    def get_model(self):
        """
        :return: Model
        """
        return self.model_

    def config_update(self, loss='mean_squared_error', activation='relu'):
        self.loss = loss
        self.activation = activation


# class Node:
#     """"
#     Node Class: each device is an object of this class
#     """

#     def __init__(self, aidi,train=1):
#         self.device_id = aidi
#         self.inp_shape = (32, 32, 3)
#         self.model = CnnModel(4)
#         self.input_tensor = self.model.add_inputs(inp_shape=self.inp_shape, num=6)
#         # self.input_tensor = keras.layers.Input(shape=self.inp_shape, name='Node'+str(self.device_id))
#         self.convp_tensor = self.model.add_convp(inputs=[self.input_tensor], parallel=0, name="base"+str(self.device_id))
#         self.model.create_model(self.input_tensor, self.convp_tensor, comp=1)
#         if train==1:
#             self.model.load("cloud")
#         # plot_model(self.model.get_model(), to_file='no_model_plot_test.png',
#         #            show_shapes=True, show_layer_names=True)

#     def calculate(self, x):
#         """
#         :param x: input List
#         :return: prediction for input vector
#         """
#         x = x.reshape((-1, 32, 32, 3))
#         return self.model.pred(x)


class CloudNet:

    def __init__(self, train):
        self.device_id = -1
        self.input = None
        self.output = None
        self.model = CnnModel(4)
        # self.model_cloud = CnnModel(4)
        self.train = train
        self.filter_num = 7
        self.kernel_size = (3, 3)
        self.pool_size = (2, 2)
        self.loss = 'categorical_crossentropy'
        self.output_num=4

        # self.node = []
        # self.input_tensor_list=[]
        # self.output_tensor_list=[]
        # for i in range(6):
        #     self.node.append(Node(i,0))
        #     in_p=self.node[i].input_tensor
        #     out_p= self.node[i].convp_tensor
        #     self.input_tensor_list.append(in_p)
        #     self.output_tensor_list.append(out_p)

        self.node_inp_shape = 32, 32, 3
        self.node_input_tensor = self.model.add_inputs(inp_shape=self.node_inp_shape, num=6)
        self.node_out_tensor = self.model.add_convp(inputs=self.node_input_tensor, parallel=0, name="base")
        
        self.node = []
        for i in range(6):
            self.node.append(keras.models.Model(self.node_input_tensor[i], self.node_out_tensor[i]))
            self.node[i].compile(loss=self.loss, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])

        self.cloud_avg_tensor = keras.layers.average(self.node_out_tensor, name="avg_layer")

        self.Cloud_inp_shape = 16, 16, self.filter_num
        # self.cloud_input_tensor = self.model.add_inputs(inp_shape=self.inp_shape, num=1, name="con_inp")
        self.cloud_input_tensor = keras.layers.Input(shape=self.Cloud_inp_shape, name="con_inp")
        self.c2 = self.model.add_convp(self.cloud_input_tensor, parallel=0, name="cloud_1st")
        self.c3 = self.model.add_convp(self.c2, parallel=0, name="cloud_2nd")

        self.flat_tensor = keras.layers.Flatten(name="flatt_layer")(self.c3)
        self.output_tensor=keras.layers.Dense(self.output_num, name="out_layer", activation="softmax")(self.flat_tensor)



        # self.output_tensor = self.model.add_fully(c3, flatten=1, name="cloud")
        self.model_cloud= keras.models.Model(self.cloud_input_tensor, self.output_tensor)
        # self.model_cloud= keras.models.Model(self.node_input_tensor[i], self.node_out_tensor[i])
        self.model_cloud.compile(loss=self.loss, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])



        # self.cloud_input_tensor=self.node_out_tensor
        # self.cloud_input_tensor=self.nod
        # self.model_cloud.create_model(self.output_tensor_list, self.output_tensor, comp=1)

        # self.model.create_model(self.input_tensor_list, self.output_tensor, comp=1)
        # self.model.create_model(self.node_input_tensor, self.model_cloud(self.node_out_tensor), comp=1)
        print((self.node_input_tensor))
        print((self.node_out_tensor))
        print(len(self.node_out_tensor))
        # self.model_all = keras.models.Model(inputs=self.node_input_tensor, outputs=self.output_tensor)
        self.model_all = keras.models.Model(input=self.node_input_tensor, output=self.model_cloud(self.cloud_avg_tensor))
        # self.model_all = keras.models.Model(input=self.node_input_tensor, output=self.output_tensor)

        self.model_all.compile(loss=self.loss, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])

        # print(self.model_all.summary)
       
        # self.model.create_model([in_p0, in_p1,in_p2,in_p3,in_p4,in_p5], self.model_cloud.create_model(), comp=1)


        if self.train == 0:
            self.model_all.load_weights("cloud_weights.h5", by_name=False)
            # self.model_all.load("cloud")

    def train_model(self, x, y, bt_s, eps):
        """
        :param x: input train vec
        :param y: output train vec
        :param bt_s: batch size
        :param eps: #epochs
        :return: None
        """
        if self.train == 1:
            y = to_categorical(y)
            # x = x['0'].reshape((-1, 32, 32, 3)), x['1'].reshape((-1, 32, 32, 3)), x['2'].reshape((-1, 32, 32, 3)),
            #      x['3'].reshape((-1, 32, 32, 3)), x['4'].reshape((-1, 32, 32, 3)), x['5'].reshape((-1, 32, 32, 3))
            # self.model.train_model(X=x, Y=y, btch_size=bt_s, ep=eps)
            # print(len(y))
            x_input=[]
            for l in range(6):
                x_input.append(x[str(l)].reshape((-1, 32, 32, 3)))

            print(len(x_input))


            history = self.model_all.fit(x=x_input, y=y, batch_size=bt_s, epochs=eps, verbose=2)
            # summarize history for accuracy
            plt.plot(history.history['acc'])
            plt.title('model accuracy')
            plt.ylabel('accuracy')
            plt.xlabel('epoch')
            plt.legend(['train'], loc='upper left')
            plt.show()
            # summarize history for loss
            plt.plot(history.history['loss'])
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train'], loc='upper left')
            plt.show()
            json = self.model_all.to_json()
            # print(json)
            # print(self.model.get_weights()[1].shape, self.model.get_weights()[2].shape)
            # print(self.model.get_weights)
            self.model_all.save_weights("cloud_weights.h5")


        else:
            raise NotImplementedError("This method is only available when training")

    def eval_comp_model(self, x, y):
        """
        :param x: input vector test
        :param y: output vector test
        :return: [loss, accuracy] for model evaluation
        """
        y = to_categorical(y)
        # x = [x['0'].reshape((-1, 32, 32, 3)), x['1'].reshape((-1, 32, 32, 3)), x['2'].reshape((-1, 32, 32, 3)),
        #      x['3'].reshape((-1, 32, 32, 3)), x['4'].reshape((-1, 32, 32, 3)), x['5'].reshape((-1, 32, 32, 3))]
        return self.model_all.evaluate(x, y)

    def eval_cloud_model(self, x, y):
        """
        :param x: input vector test
        :param y: output vector test
        :return: [loss, accuracy] for model evaluation
        """
        y = to_categorical(y)
        # x = [x['0'].reshape((-1, 32, 32, 3)), x['1'].reshape((-1, 32, 32, 3)), x['2'].reshape((-1, 32, 32, 3)),
        #      x['3'].reshape((-1, 32, 32, 3)), x['4'].reshape((-1, 32, 32, 3)), x['5'].reshape((-1, 32, 32, 3))]
        return self.model_cloud.evaluate(x, y)




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


        return np.array(avg_batch)

    def calculate_claud(self, x_cl, action=0,apply_action=1):
        avg_inp=self.calc_avg_cloud(x_cl, action, apply_action)
                        
        return self.model_cloud.predict(avg_inp)

    def evaluate_claud(self, x_cl, y, action,apply_action=1):

        y_cat = to_categorical(y)
        avg_inp=self.calc_avg_cloud(x_cl, action, apply_action)
                        
        return self.model_cloud.evaluate(avg_inp,y_cat)


    # def get_in_out_tensor(self):
    #     return self.input_tensor, self.output_tensor
# 
