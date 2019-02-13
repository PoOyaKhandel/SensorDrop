from cnnmodel import CnnModel


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
        # print(self.model.model.get_weights()[6].shape)
        # print(self.model.model.get_weights())
        # plot_model(self.model.get_model(), to_file='no_model_plot_test.png',
        #            show_shapes=True, show_layer_names=True)

    def calculate(self, x):
        """
        :param x: input List
        :return: prediction for input vector
        """
        x = x.reshape((-1, 32, 32, 3))
        return self.model.pred(x)
