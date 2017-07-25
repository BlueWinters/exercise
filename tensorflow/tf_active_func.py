# coding=utf-8

import tensorflow as tf

def active_func_logging():
    sess = tf.Session()
    A = tf.Variable(name='A', initial_value=[[1,2,3],[3,2,1]], dtype=tf.float32)
    logA = tf.log(A)
    sess.run(tf.global_variables_initializer())

    print(sess.run(A))
    print(sess.run(logA))

    sess.close()

    # 输出
    # [[ 1.  2.  3.]
    #  [ 3.  2.  1.]]
    # [[ 0.          0.69314718  1.09861231]
    #  [ 1.09861231  0.69314718  0.        ]]


if __name__ == '__main__':
    active_func_logging()
    pass
