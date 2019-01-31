import keras.models
import keras.optimizers
import keras.layers
import keras
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from keras.utils.vis_utils import plot_model


class CnnModel:
    filter_num = 7

    def __init__(self):
        self.model = None
        self.optimizer = None
        self.loss = 'mean_squared_error'
        self.activation = 'relu'
        self.kernel_size = (3, 3)
        self.filter_num = CnnModel.filter_num
        self.input_shape = (32, 32, 3)
        self.pool_size = (2, 2)
        self.dense_len = 4  # person, bus, car, not-present

    def define_convp(self, convp_in):
        """
        :param convp_in: list of input layer for convp blocks
        :return: Convp Block output
        """
        conv2d_base = self.__define_conv2d()
        pooling_base = self.__define_max_pool()
        batch_norm_base = self.__define_batch_normalization()
        output = []

        for n in range(len(convp_in)):
            output.append(batch_norm_base(pooling_base(conv2d_base(convp_in[n]))))

        return output

    def __define_conv2d(self):
        """
        :return: conv2d Layer
        """
        return keras.layers.Conv2D(self.filter_num, self.kernel_size, strides=(1, 1), activation=self.activation,
                                   padding='same',
                                   kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=1))

    def __define_max_pool(self):
        """
        :return: pooling Layer
        """
        return keras.layers.MaxPooling2D(pool_size=self.pool_size, padding='same')

    def __define_batch_normalization(self):
        """
        :return: Batch Layer
        """
        return keras.layers.BatchNormalization()

    def __define_flatten(self):
        """
        :return: Flatten Layer
        """
        return keras.layers.Flatten()

    def __define_fully(self):
        """
        :return: Fully Layer
        """
        return keras.layers.Dense(self.dense_len)

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

    def create_model(self, inp, out):
        """
        defining model with Functional API keras
        :param inp: input layer of model
        :param out: output layer of model
        :return: None
        """
        self.model = keras.models.Model(inputs=inp, outputs=out)
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
        print(json)
        print(self.model.get_weights()[1].shape, self.model.get_weights()[2].shape)
        # self.model.save_weights("D:\Library\Statistical Learning\SensorDrop\w.h5")

    def eval_model(self, X, Y):
        """
        Evaluate trained model
        :param X: test input
        :param Y: test output
        :return: [loss, accuracy]
        """
        return self.model.evaluate(X, Y)

    def add_inputs(self, num, inp_shape):
        """
        :param num: number of inputs
        :param inp_shape: shape of each input
        :return: inputs Tensor
        """
        inputs = []
        if num != 0:
            for n in range(num):
                inputs.append(keras.layers.Input(shape=inp_shape))
        else:
            raise NotImplementedError("num can not be zero")
        return inputs

    def add_convp(self, inputs, parallel=0):
        """
        :param inputs: gets inputs of set(s) of Convp blocks
        :param parallel: is 1 if parallel blocks desired
        :return: concatenated block of parallel convps
        """
        if parallel == 1:
            conv_out = self.define_convp(inputs)
            return keras.layers.concatenate(conv_out)
        else:
            conv_out = self.define_convp([inputs])
            return conv_out[0]

    def add_fully(self, fully_in, flatten=1):
        """
        :param fully_in: fully block input tensor
        :param flatten: if 1 also add flatten layer before fully layer
        :return:
        """
        if flatten == 1:
            flat = self.__define_flatten()(fully_in)
            return self.__define_fully()(flat)
        else:
            return self.__define_fully(fully_in)
    
    def get_model(self):
        return self.model


class Node:
    """"
    Node Class: each device is an object of this class
    """

    def __init__(self, aidi):
        self.device_id = aidi
        self.inp_shape = (3, 32, 32)
        self.input = None
        self.output = None
        self.model = CnnModel()
        self.model.__compile_model()


class CloudNet:

    def __init__(self, train):
        self.device_id = -1
        self.input = None
        self.output = None
        self.model = CnnModel()
        self.train = train
        self.complexity = 2
        if self.train == 1:
            self.inp_shape = 32, 32, 3
            input_layer = self.model.add_inputs(num=6, inp_shape=self.inp_shape)
            concat_layer = self.model.add_convp(inputs=input_layer, parallel=1)
            print(concat_layer)
            c2 = self.model.add_convp(concat_layer)
            print("c2", c2)
            c3 = self.model.add_convp(c2)
            print("c3", c3)
            output_layer = self.model.add_fully(c3, flatten=1)
            print(output_layer)
            self.model.create_model(input_layer, output_layer)
            # plot_model(self.model.get_model(), to_file='model_plot.png', show_shapes=True, show_layer_names=True)
            # self.model.compile_model()
        else:
            self.inp_shape = (CnnModel.filter_num, 3, 32, 32)
            self.complexity = 2

    def train_model(self, x, y, bt_s, eps):
        if self.train == 1:
            y = to_categorical(y)
            x = [x['0'].reshape((-1, 32, 32, 3)), x['1'].reshape((-1, 32, 32, 3)), x['2'].reshape((-1, 32, 32, 3)),
                 x['3'].reshape((-1, 32, 32, 3)), x['4'].reshape((-1, 32, 32, 3)), x['5'].reshape((-1, 32, 32, 3))]
            self.model.train_model(X=x, Y=y, btch_size=bt_s, ep=eps)
        else:
            raise NotImplementedError("This method is only available when training")

    def eval_model(self, x, y):
        y = to_categorical(y)
        x = [x['0'].reshape((-1, 32, 32, 3)), x['1'].reshape((-1, 32, 32, 3)), x['2'].reshape((-1, 32, 32, 3)),
             x['3'].reshape((-1, 32, 32, 3)), x['4'].reshape((-1, 32, 32, 3)), x['5'].reshape((-1, 32, 32, 3))]
        return self.model.eval_model(x, y)
