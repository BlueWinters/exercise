# coding=utf-8

import tensorflow as tf


def operator_matmul():
    x = tf.get_variable(name='x', shape=[10, 4])
    W = tf.get_variable(name='A', shape=[4, 3])
    b = tf.get_variable(name='B', shape=[3])

    # y = W*x + b
    y = tf.matmul(x, W) + b
    print(y.shape)

    # 输出
    # (10,3)

def operator_mult_variable_same_size():
    sess = tf.Session()
    A = tf.Variable(name='A', initial_value=[[1,2,3],[3,2,1]])
    B = tf.Variable(name='B', initial_value=[[6,4,5],[4,5,6]])
    sess.run(tf.global_variables_initializer())

    C1 = A * B
    C2 = tf.multiply(A, B)
    C3 = 1 - A
    C4 = 3 * A

    print(sess.run(A))
    print(sess.run(B))
    print(sess.run(C1))
    print(sess.run(C2))
    print(sess.run(C3))
    print(sess.run(C4))

    sess.close()

    # 输出
    # [[1 2 3]
    #  [3 2 1]]
    # [[6 4 5]
    #  [4 5 6]]
    # [[ 6  8 15]
    #  [12 10  6]]
    # [[ 6  8 15]
    #  [12 10  6]]
    # [[ 0 -1 -2]
    #  [-2 -1  0]]
    # [[3 6 9]
    #  [9 6 3]]

def operator_multiply_random_variable():
    sess = tf.Session()
    A = tf.Variable(initial_value=tf.random_normal(shape=[2,3]))
    B = tf.Variable(initial_value=tf.random_normal(shape=[2,4]))
    B1 = tf.reduce_sum(B, axis=1, keep_dims=True)
    sess.run(tf.global_variables_initializer())

    C1 = A * B1
    C2 = tf.multiply(A, B1)
    C3 = 1 / A

    print(A.shape)
    print(B1.shape)
    print(sess.run(A))
    print(sess.run(B1))
    print(sess.run(C1))
    print(sess.run(C2))
    print(sess.run(C3))

    sess.close()

    # 随机输出

def operator_multiply_variable():
    sess = tf.Session()
    A = tf.Variable(name='A', initial_value=[[1,2,3],[4,5,6]])
    B = tf.Variable(name='B', initial_value=[[1],[2]])
    sess.run(tf.global_variables_initializer())

    C1 = A * B
    C2 = tf.multiply(A, B)

    print(A.shape)
    print(B.shape)
    print(sess.run(A))
    print(sess.run(B))
    print(sess.run(C1))
    print(sess.run(C2))

    sess.close()

    # 输出
    # (2, 3)
    # (2, 1)
    # A:
    # [[1 2 3]
    #  [4 5 6]]
    # B:
    # [[1]
    #  [2]]
    # C1:
    # [[ 1  2  3]
    #  [ 8 10 12]]
    # C2:
    # [[ 1  2  3]
    #  [ 8 10 12]]

def operator_reduce_sum():
    x = tf.get_variable(name='x', shape=[5, 10])
    x_axis0 = tf.reduce_sum(x, axis=0)
    x_axis1 = tf.reduce_sum(x, axis=1)
    x_axis1_keepdim = tf.reduce_sum(x, axis=1, keep_dims=True)

    print(x_axis0.shape)
    print(x_axis1.shape)
    print(x_axis1_keepdim.shape)

    # 输出
    # (10,)
    # (5,)
    # (5,1)


if __name__ == '__main__':
    # operator_matmul()
    # operator_reduce_sum()
    # operator_multiply_variable()
    operator_multiply_random_variable()
    # operator_mult_variable_same_size()
    pass