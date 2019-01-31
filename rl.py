import keras.models
import keras.optimizers
import keras.layers
import keras
from node import CnnModel
from scipy.stats import bernoulli


class PolicyNetwork:
    def __init__(self):
        self.pnet = CnnModel(d_size=6)
        self.rl = RL()
        self.alpha = 0.8
        self.inp_shape = 16, 16, 42
        input_tensor = self.pnet.add_inputs(inp_shape=self.inp_shape, num=6, name="pnet_input")
        convp_tensor = self.pnet.add_convp(inputs=input_tensor, parallel=-1, name="pnet_conv_1st")
        convp2_tensor = self.pnet.add_convp(inputs=convp_tensor, parallel=0, name="pnet_conv_2nd")
        output_tensor = self.pnet.add_fully(convp2_tensor, flatten=1, name="pnet_fully")
        self.pnet.create_model(input_tensor, output_tensor, comp=1)

    def train(self, x, u, pred):
        # reward = self.rl.reward(u.count(1),pred)
        pass




    def calculate(self, x):
        # x.resahpe()
        return self.pnet.pred(x)


class RL:
    def __init__(self):
        self.reward_minus_const = -0.1
        self.device_count = 6
        pass
    
    def reward(self, device_n, prediction):
        if prediction:
            return 1 - (device_n/self.device_count)**2
        else:
            return self.reward_minus_const


    def get_loss_function(self):
        pass

    def bernoulli(self, s):
        pass


