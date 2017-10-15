# coding=utf-8

import tensorflow as tf

def reduce_mean():
    A = tf.get_variable('A', [10,28,28,32])
    B = tf.reduce_mean(A, axis=[1,2])

    print(A.shape)
    print(B.shape)


if __name__ == '__main__':
    reduce_mean()
    pass
