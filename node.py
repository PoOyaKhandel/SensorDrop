import keras.models
import keras.optimizers
import keras.layers
import keras
from keras.utils import to_categorical


class CnnModel:
    def __int__(self):
        self.model = keras.models.Sequential()
        self.optimizer = None
        self.loss = 'mse'
        self.activation = 'relu'
        self.kernel_size = (3, 3)
        self.filter_num = 7
        self.input_shape = (3, 32, 32)
        self.pool_size = (2, 2)
        self.dense_len = 4  # person, bus, car, not-present

    def conv2d_define(self):
        return keras.layers.Conv2D(self.filter_num, self.kernel_size, strides=(1, 1), input_shape=self.input_shape,
                                   activation=self.activation)

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

    def compile_model(self, loss=self.loss):
        self.optimizer_conf()
        self.model.compile(loss=loss, optimizer=self.optimizer)


class Node:
    """"
    Node Class: each device is an object of this class
    """
    def __int__(self, aidi):
        self.device_id = aidi
        self.model = CnnModel()
        self.model.add_layers_convp()
        self.model.compile_model()


class CloudNet:
    def __int__(self):
        self.device_id = -1
        self.model = CnnModel()
        self.complexity = 2
        for _ in range(self.complexity):
            self.model.add_layers_convp()
            self.model.add_layers_convp()
        self.model.add_layers_fully()
        self.model.compile_model()




















