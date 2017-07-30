# coding=utf-8

import tensorflow as tf

def concat_to_variable():
    A = tf.get_variable(name='A', shape=[2,3])
    B = tf.get_variable(name='B', shape=[2,5])
    C = tf.get_variable(name='C', shape=[4,3])

    # axis参数用于指定想要进行concat的那个维度
    A_C = tf.concat(values=[A, C], axis=0)
    A_B = tf.concat(values=[A, B], axis=1)


    print(A_B.shape)
    print(A_C.shape)

    # 输出
    # (2, 8)
    # (6, 3)

def slice_to_variable():
    A = tf.get_variable(name='A', shape=[6,8])

    # 切割数据
    A1 = tf.slice(A, [0,0], [2,-1])
    A2 = tf.slice(A, [3,0], [3,-1])

    print(A1.shape)
    print(A2.shape)

    shape = tf.shape(A)
    s1 = shape[0]

    # 输出：
    # (2, 8)
    # (3, 8)


if __name__ == '__main__':
    # concat_to_variable()
    slice_to_variable()