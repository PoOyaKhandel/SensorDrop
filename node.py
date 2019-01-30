import keras.models
import keras.optimizers
import keras.layers
import keras
from keras.utils import to_categorical
import matplotlib.pyplot as plt



class CnnModel:
    filter_num = 7

    def __init__(self):
        self.model = keras.models.Sequential()
        self.optimizer = None
        self.loss = 'binary_crossentropy'
        self.activation = 'relu'
        self.kernel_size = (3, 3)
        self.filter_num = CnnModel.filter_num
        self.input_shape = (32, 32, 3)
        self.pool_size = (2, 2)
        self.dense_len = 4  # person, bus, car, not-present

    def conv2d_define(self):
        return keras.layers.Conv2D(self.filter_num, self.kernel_size, strides=(1, 1), activation=self.activation,
                                   padding='same',
                                   kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=1))

    def max_pool_define(self):
        return keras.layers.MaxPooling2D(pool_size=self.pool_size, padding='same')

    def linear_define(self):
        return keras.layers.Dense((32, 32, 3),
                                  kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=1))

    def flatten_define(self):
        return keras.layers.Flatten()

    def bath_normalization_define(self):
        return keras.layers.BatchNormalization()  # parameter config !

    def add_layers_convp(self):
        self.model.add(self.conv2d_define())
        self.model.add(self.max_pool_define())
        self.model.add(self.bath_normalization_define())

    def fully_define(self):
        return keras.layers.Dense(self.dense_len)

    def add_layers_fully(self):
        self.model.add(keras.layers.Flatten())
        self.model.add(self.fully_define())
        self.model.add(self.bath_normalization_define())


    def optimizer_conf(self, lr=0.001, beta_1=0.9, beta_2=0.999):
        self.optimizer = keras.optimizers.Adam(lr, beta_1, beta_2, epsilon=1e-8)

    def compile_model(self):
        self.optimizer_conf()
        self.model.compile(loss=self.loss, optimizer=self.optimizer, metrics=['accuracy'])

    def create_input(self, inp_shape):
        self.model.add(keras.layers.InputLayer(input_shape=inp_shape))

    def train_model(self, X, Y, btch_size, ep):
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

    def eval_model(self, X, Y):
        return self.model.evaluate(X, Y)

    def define_model(self, inp, out):
        self.model = keras.models.Model(inputs=inp, outputs=out)

    def define_inputs(self, num, inp_shape):
        inputs = []
        for n in range(num):
            inputs.append(keras.layers.Input(shape=inp_shape))

        return inputs

    def add_parallel_convp(self, num, inputs):
        x = []
        print(inputs)
        for n in range(num):
            x.append((self.conv2d_define())(inputs[n]))

        print(x)
        return keras.layers.concatenate(x)



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
        self.create_input(self.inp_shape)
        self.model.add_layers_convp()
        self.model.compile_model()

    # def train_model(self, x, y, bt_s, eps):
    #     y = to_categorical(y)
    #     self.model.train_m(X=x, Y=y, btch_size=bt_s, ep=eps)


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
            input_layer = self.model.define_inputs(num=6, inp_shape=self.inp_shape)
            concat_layer = self.model.add_parallel_convp(num=6, inputs=input_layer)
            print(concat_layer)
            # flatten_layer0 = self.model.flatten_define()(concat_layer)
            # flatten_layer0 = keras.layers.Reshape((3, 32, 32))(concat_layer)
            # print((flatten_layer0))
            # linear_layer = self.model.linear_define()(concat_layer)
            c2 = self.model.conv2d_define()(concat_layer)
            print("c2", c2)
            c3 = self.model.conv2d_define()(c2)
            print("c3", c3)
            flatten_layer1 = self.model.flatten_define()(c3)
            print(flatten_layer1)
            output_layer = self.model.fully_define()(flatten_layer1)
            print(output_layer)
            self.model.define_model(input_layer, output_layer)
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
        x = [x['0'].reshape((-1, 32, 32, 3)), x['1'].reshape((-1, 32, 32, 3)), x['2'].reshape((-1, 32, 32, 3)),
             x['3'].reshape((-1, 32, 32, 3)), x['4'].reshape((-1, 32, 32, 3)), x['5'].reshape((-1, 32, 32, 3))]
        self.model.train_model(X=x, Y=y, btch_size=bt_s, ep=eps)
        # else:
        #     print("Not in training mode")

    def eval_model(self, x, y):
        y = to_categorical(y)
        x = [x['0'].reshape((-1, 32, 32, 3)), x['1'].reshape((-1, 32, 32, 3)), x['2'].reshape((-1, 32, 32, 3)),
             x['3'].reshape((-1, 32, 32, 3)), x['4'].reshape((-1, 32, 32, 3)), x['5'].reshape((-1, 32, 32, 3))]
        return self.model.eval_model(x, y)

