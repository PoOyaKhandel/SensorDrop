import keras.models
import keras.optimizers
import keras.layers
import keras
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from keras.utils.vis_utils import plot_model
import numpy as np


class CnnModel:
    filter_num = 7
    weightPath = "w.h5"

    def __init__(self, d_size):
        self.model = None
        self.optimizer = None
        self.loss = 'mean_squared_error'
        self.activation = 'sigmoid'
        self.kernel_size = (3, 3)
        self.filter_num = CnnModel.filter_num
        self.input_shape = (32, 32, 3)
        self.pool_size = (2, 2)
        self.dense_len = d_size  # person, bus, car, not-present

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

        for n in range(len(convp_in)):
            output.append(batch_norm_base(pooling_base(conv2d_base(convp_in[n]))))

        return output

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
        return keras.layers.Dense(self.dense_len, name=name, activation=self.activation)

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
        self.model.compile(loss=self.loss, optimizer=self.optimizer, metrics=['accuracy'])

    def create_model(self, inp, out, comp):
        """
        defining model with Functional API keras
        :param comp: if 1 compile the model
        :param inp: input layer of model
        :param out: output layer of model
        :return: None
        """
        self.model = keras.models.Model(inputs=inp, outputs=out)
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
        history = self.model.fit(x=X, y=Y, batch_size=btch_size, epochs=ep, verbose=2)
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
        json = self.model.to_json()
        # print(json)
        # print(self.model.get_weights()[1].shape, self.model.get_weights()[2].shape)
        # print(self.model.get_weights)
        self.model.save_weights("cloud_weights.h5")

    def eval_model(self, X, Y):
        """
        Evaluate trained model
        :param X: test input
        :param Y: test output
        :return: [loss, accuracy]
        """
        return self.model.evaluate(X, Y)

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

    def add_convp(self, inputs, parallel=0, name="?"):
        """
        :param name: convp blocks name
        :param inputs: gets inputs of set(s) of Convp blocks
        :param parallel: 1-> parallel convp + concatenate. 0-> single convp. -1-> concatenate+single convp
        :return: concatenated block of parallel convps
        """
        if parallel == 1:
            conv_out = self.__define_convp(inputs, name)
            return keras.layers.concatenate(conv_out, name="concat")
        elif parallel == 0:
            conv_out = self.__define_convp(inputs, name)
            return conv_out[0]
        elif parallel == -1:
            concat = keras.layers.concatenate(inputs, name="concat")
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
        self.model.load_weights(weights+"_weights.h5", by_name=True)

    def pred(self, x):
        """
        :param x: input vector
        :return: model prediction of x
        """
        return self.model.predict(x)

    def get_model(self):
        """
        :return: Model
        """
        return self.model

    def config_update(self, loss='mean_squared_error', activation='relu'):
        self.loss = loss
        self.activation = activation


class Node:
    """"
    Node Class: each device is an object of this class
    """

    def __init__(self, aidi):
        self.device_id = aidi
        self.inp_shape = (32, 32, 3)
        self.model = CnnModel(4)
        input_tensor = self.model.add_inputs(inp_shape=self.inp_shape, num=1)
        convp_tensor = self.model.add_convp(inputs=input_tensor, parallel=0, name="base")
        self.model.create_model(input_tensor, convp_tensor, comp=1)
        self.model.load("cloud")
        # plot_model(self.model.get_model(), to_file='no_model_plot_test.png',
        #            show_shapes=True, show_layer_names=True)

    def calculate(self, x):
        """
        :param x: input List
        :return: prediction for input vector
        """
        x = x.reshape((-1, 32, 32, 3))
        return self.model.pred(x)


class CloudNet:

    def __init__(self, train):
        self.device_id = -1
        self.input = None
        self.output = None
        self.model = CnnModel(4)
        self.train = train
        if self.train == 1:
            self.inp_shape = 32, 32, 3
            self.input_tensor = self.model.add_inputs(inp_shape=self.inp_shape, num=6)
            concat_tensor = self.model.add_convp(inputs=self.input_tensor, parallel=1, name="base")
            c2 = self.model.add_convp([concat_tensor], parallel=0, name="cloud_1st")
            c3 = self.model.add_convp([c2], parallel=0, name="cloud_2nd")
            self.output_tensor = self.model.add_fully(c3, flatten=1, name="cloud")
            self.model.create_model(self.input_tensor, self.output_tensor, comp=1)
        else:
            self.inp_shape = 16, 16, 7
            self.input_tensor = self.model.add_inputs(inp_shape=self.inp_shape, num=6, name="con_inp")
            c2 = self.model.add_convp(self.input_tensor, parallel=-1, name="cloud_1st")
            c3 = self.model.add_convp(c2, parallel=0, name="cloud_2st")
            self.output_tensor = self.model.add_fully(c3, flatten=1, name="cloud")
            self.model.create_model(self.input_tensor, self.output_tensor, comp=1)
            self.model.load("cloud")

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
            x = [x['0'].reshape((-1, 32, 32, 3)), x['1'].reshape((-1, 32, 32, 3)), x['2'].reshape((-1, 32, 32, 3)),
                 x['3'].reshape((-1, 32, 32, 3)), x['4'].reshape((-1, 32, 32, 3)), x['5'].reshape((-1, 32, 32, 3))]
            self.model.train_model(X=x, Y=y, btch_size=bt_s, ep=eps)
        else:
            raise NotImplementedError("This method is only available when training")

    def eval_model(self, x, y):
        """
        :param x: input vector test
        :param y: output vector test
        :return: [loss, accuracy] for model evaluation
        """
        y = to_categorical(y)
        # x = [x['0'].reshape((-1, 32, 32, 3)), x['1'].reshape((-1, 32, 32, 3)), x['2'].reshape((-1, 32, 32, 3)),
        #      x['3'].reshape((-1, 32, 32, 3)), x['4'].reshape((-1, 32, 32, 3)), x['5'].reshape((-1, 32, 32, 3))]
        return self.model.eval_model(x, y)

    def calculate(self, x, policy):
        # x = [x['0'].reshape((-1, 32, 32, 3)), x['1'].reshape((-1, 32, 32, 3)), x['2'].reshape((-1, 32, 32, 3)),
        #      x['3'].reshape((-1, 32, 32, 3)), x['4'].reshape((-1, 32, 32, 3)), x['5'].reshape((-1, 32, 32, 3))]
        print(x[0].shape)
        zer = np.zeros_like(x[0])

        for i in range(policy.shape[1]):
            for n in range(x[0].shape[0]):
                if policy[n, i] == 0:
                    x[i, n] = zer

        return self.model.pred(x)

    def get_in_out_tensor(self):
        return self.input_tensor, self.output_tensor
