from node import CnnModel
import numpy as np
from keras.utils import to_categorical

class CloudNet:

    def __init__(self, train):
        self.device_id = -1
        self.input = None
        self.output = None
        self.model = CnnModel(4)
        self.train = train
        self.filter_num = 2
        if self.train == 1:
            self.inp_shape = 32, 32, 3
            self.input_tensor = self.model.add_inputs(inp_shape=self.inp_shape, num=6)
            concat_tensor = self.model.add_convp(inputs=self.input_tensor, parallel=1, name="base")
            c2 = self.model.add_convp([concat_tensor], parallel=0, name="cloud_1st")
            c3 = self.model.add_convp([c2], parallel=0, name="cloud_2nd")
            self.output_tensor = self.model.add_fully(c3, flatten=1, name="cloud")
            self.model.create_model(self.input_tensor, self.output_tensor, comp=1)
        else:
            self.inp_shape = 16, 16, self.filter_num
            self.input_tensor = self.model.add_inputs(inp_shape=self.inp_shape, num=6, name="con_inp")
            c2 = self.model.add_convp(self.input_tensor, parallel=-1, name="cloud_1st")
            c3 = self.model.add_convp(c2, parallel=0, name="cloud_2nd")
            self.output_tensor = self.model.add_fully(c3, flatten=1, name="cloud")
            self.model.create_model(self.input_tensor, self.output_tensor, comp=1)
            self.model.load("cloud")
            # print(self.model.model.get_weights())

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
        x = [x['0'].reshape((-1, 32, 32, 3)), x['1'].reshape((-1, 32, 32, 3)), x['2'].reshape((-1, 32, 32, 3)),
             x['3'].reshape((-1, 32, 32, 3)), x['4'].reshape((-1, 32, 32, 3)), x['5'].reshape((-1, 32, 32, 3))]
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
