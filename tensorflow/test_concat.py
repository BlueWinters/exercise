# coding=utf-8

import tensorflow as tf

def concat_tow_variable():
    A = tf.get_variable(name='A', shape=[2,3])
    B = tf.get_variable(name='B', shape=[2,5])
    C = tf.get_variable(name='C', shape=[4,3])

    # axis参数用于指定想要进行concat的那个维度
    A_C = tf.concat(values=[A, C], axis=0)
    A_B = tf.concat(values=[A, B], axis=1)


    print(A_B.shape)
    print(A_C.shape)


if __name__ == '__main__':
    concat_tow_variable()