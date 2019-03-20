from dataset import datasets
import matplotlib.pyplot as plt
import argparse
import numpy as np
from keras.utils import to_categorical

x1, x2, y1, y2 = datasets.get_mvmc(te_percent=0.20)

def input_dict_to_list(x,y):
    y_ = to_categorical(y)
    x_ = [x['0'].transpose(0, 2, 3, 1), x['1'].transpose(0, 2, 3, 1), x['2'].transpose(0, 2, 3, 1),
            x['3'].transpose(0, 2, 3, 1), x['4'].transpose(0, 2, 3, 1), x['5'].transpose(0, 2, 3, 1)]
    return x_,y_

def show_img(imgs, idx, action):
    fig = plt.figure()
    for i, img in zip(range(6), [im[idx] for im in imgs]):
        ax = fig.add_subplot(1, 6, i+1)
        if action[i] == 0:
            ax.set_title('drop')
        else:
            ax.set_title('keep')
        ax.imshow(img)
    fig.savefig("./output/fig"+str(idx))

if __name__ == "__main__":
    index = 10
    a, _ = input_dict_to_list(x1, y1)
    show_img(a, index, [0, 1, 0, 1, 0, 1])

