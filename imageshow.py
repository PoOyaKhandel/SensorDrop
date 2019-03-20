from dataset import datasets
import matplotlib.pyplot as plt
import argparse


x1, x2, y1, y2 = datasets.get_mvmc(te_percent=0.20)


def show_img(obj, index, action):
    fig = plt.figure()
    for i in range(6):
        ax = fig.add_subplot(1, 6, i+1)
        if action[i] == 0:
            ax.set_title('drop')
        else:
            ax.set_title('keep')
        ax.imshow(obj[str(i)][index].transpose(1, 2, 0))
    plt.show()


show_img(x1, 1, [0, 1, 0, 1, 0, 1])

