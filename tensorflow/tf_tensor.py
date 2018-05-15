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

def sub_operator():
    var1 = tf.convert_to_tensor([1, 2, 3], dtype=tf.float32, name='var1')
    var2 = tf.convert_to_tensor([1, 2, 3], dtype=tf.float32, name='var2')

    var1 = tf.expand_dims(var1, axis=0)
    var2 = tf.expand_dims(var2, axis=1)

    print(var1.shape)
    print(var2.shape)

    with tf.Session() as sess:
        print(sess.run(var1))
        print(sess.run(var2))
        print(sess.run(var1 - var2))
        print(sess.run(var2 - var1))

def range_tensor():
    var1 = tf.range(10)
    var2 = var1 * 2

    with tf.Session() as sess:
        print(sess.run(var1))
        print(sess.run(var2))

    # 输出
    # [0 1 2 3 4 5 6 7 8 9]
    # [0  2  4  6  8 10 12 14 16 18]

if __name__ == '__main__':
    # access_tensor()
    # sub_operator()
    range_tensor()