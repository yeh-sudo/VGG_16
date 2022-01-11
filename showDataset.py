import numpy as np
import cv2
import matplotlib
import random
import matplotlib.pyplot as plt


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dic = pickle.load(fo, encoding='bytes')
    return dic


def showData():
    classes = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    data = unpickle('./dataset/cifar-10-batches-py/data_batch_1')
    imageType = []
    img = []
    for i in range(0, 9):
        idx = random.randint(1, 10000)
        imageType.append(classes[data[b'labels'][idx-1]])
        tmp = data[b'data'][idx-1]
        r = np.zeros((32, 32), dtype='uint8')
        g = np.zeros((32, 32), dtype='uint8')
        b = np.zeros((32, 32), dtype='uint8')
        cnt = 0
        while cnt < 32:
            r[cnt] = tmp[cnt * 32: cnt * 32 + 32]
            cnt += 1
        while cnt < 64:
            g[cnt-32] = tmp[cnt * 32: cnt * 32 + 32]
            cnt += 1
        while cnt < 96:
            b[cnt-64] = tmp[cnt * 32: cnt * 32 + 32]
            cnt += 1
        merge_img = cv2.merge([b, g, r])
        img.append(merge_img)
    fig = plt.figure()
    axes = []
    for i in range(9):
        b = img[i]
        axes.append(fig.add_subplot(3, 3, i + 1))
        subplot_title = imageType[i]
        axes[-1].set_title(subplot_title)
        plt.imshow(b)
        plt.axis('off')
    fig.tight_layout()
    plt.show()

