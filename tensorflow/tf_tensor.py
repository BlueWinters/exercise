# coding=utf-8

import tensorflow as tf


def access_tensor():
    var = tf.get_variable(name='var', shape=[3,4])
    var1 = var[:,:2]
    var2 = var[1:-1,:]

    print(var1.shape)
    print(var2.shape)

    # 输出
    # (3, 2)
    # (1, 4)

if __name__ == '__main__':
    access_tensor()