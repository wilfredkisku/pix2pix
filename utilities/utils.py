import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import matplotlib.pyplot as plt
import tensorflow as tf

class preprocessing:
    def __init__(self):

        self.BUFFER_SIZE = 400
        self.BATCH_SIZE = 32
        self.IMG_WIDTH = 256
        self.IMG_HEIGHT = 256
        self.PATH = '/home/wilfred/Datasets/facades/'

    def load(self,image_file):

        image = tf.io.read_file(image_file)
        image = tf.image.decode_jpeg(image)

        w = tf.shape(image)[1]

        w = w // 2

        real_image  = image[ :, :w, :]
        input_image = image[ :, w:, :]

        input_image = tf.cast(input_image, tf.float32)
        real_image = tf.cast(real_image, tf.float32)

        return input_image, real_image

if __name__ == "__main__":

    pre = preprocessing()
    inp, re = pre.load(pre.PATH+'/train/100.jpg')
    plt.figure(num=1)
    plt.imshow(inp/255.)
    plt.axis('off')

    plt.figure(num=2)
    plt.imshow(re/255.)
    plt.axis('off')
    plt.show()
