import keras.models
import keras.optimizers
import keras.layers
import keras
from keras.utils import to_categorical


class CnnModel:
    filter_num = 7

    def __int__(self):
        self.model = keras.models.Sequential()
        self.optimizer = None
        self.loss = 'binary_cross_entropy'
        self.activation = 'relu'
        self.kernel_size = (3, 3)
        self.filter_num = CnnModel.filter_num
        self.input_shape = (3, 32, 32)
        self.pool_size = (2, 2)
        self.dense_len = 4  # person, bus, car, not-present

    def conv2d_define(self):
        return keras.layers.Conv2D(self.filter_num, self.kernel_size, strides=(1, 1), activation=self.activation)

    def max_pool_define(self):
        return keras.layers.MaxPool2D(self.pool_size)

    def bath_normalization_define(self):
        return keras.layers.BatchNormalization()  # parameter config !

    def add_layers_convp(self):
        self.model.add(self.conv2d_define())
        self.model.add(self.max_pool_define())
        self.model.add(self.bath_normalization_define())

    def fully_define(self):
        return keras.layers.Dense(self.dense_len)

    def add_layers_fully(self):
        self.model.add(self.fully_define())
        self.model.add(self.bath_normalization_define())


    def optimizer_conf(self, lr=0.001, beta_1=0.9, beta_2=0.999):
        self.optimizer = keras.optimizers.Adam(lr, beta_1, beta_2)

    def compile_model(self):
        self.optimizer_conf()
        self.model.compile(loss=self.loss, optimizer=self.optimizer)

    def create_input(self, inp_shape):
        self.model.add(keras.engine.input_layer.Input(shape=inp_shape))

    def train_model(self, X, Y, btch_size, ep):
        self.model.fit(x=X, y=Y, batch_size=btch_size, epochs=ep)


class Node:
    """"
    Node Class: each device is an object of this class
    """
    def __int__(self, aidi):
        self.device_id = aidi
        self.inp_shape = (3, 32, 32)
        self.input = None
        self.output = None
        self.model = CnnModel()
        self.create_input(self.inp_shape)
        self.model.add_layers_convp()
        self.model.compile_model()

    # def train_model(self, x, y, bt_s, eps):
    #     y = to_categorical(y)
    #     self.model.train_m(X=x, Y=y, btch_size=bt_s, ep=eps)


class CloudNet:

    def __int__(self):
        self.device_id = -1
        self.input = None
        self.output = None
        self.model = CnnModel()
        self.train = 1
        if self.train == 1:
            self.inp_shape = (3, 32, 32)
            self.complexity = 3
            self.create_input(self.inp_shape)
            for _ in range(self.complexity):
                self.model.add_layers_convp()
            self.model.add_layers_fully()
            self.model.compile_model()

        else:
            self.inp_shape = (CnnModel.filter_num, 3, 32, 32)
            self.complexity = 2
            self.create_input(self.inp_shape)
            for _ in range(self.complexity):
                self.model.add_layers_convp()
            self.model.add_layers_fully()
            self.model.compile_model()

    def train_model(self, x, y, bt_s, eps):
        # if self.train == 1:
        y = to_categorical(y)
        self.model.train_m(X=x, Y=y, btch_size=bt_s, ep=eps)
        # else:
        #     print("Not in training mode")



















