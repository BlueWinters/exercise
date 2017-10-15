# coding=utf-8

import tensorflow as tf

def get_top_k():
    x = tf.constant(-1.0, shape=[100,50])

    k = 5
    values, indices = tf.nn.top_k(x, k)

    # We need to create full indices like [[0, 0], [0, 1], [1, 2], [1, 1]]
    my_range = tf.expand_dims(tf.range(0, indices.get_shape()[0]), 1)  # will be [[0], [1]]
    my_range_repeated = tf.tile(my_range, [1, k])  # will be [[0, 0], [1, 1]]

    # change shapes to [N, k, 1] and [N, k, 1], to concatenate into [N, k, 2]
    full_indices = tf.concat([tf.expand_dims(my_range_repeated, 2), tf.expand_dims(indices, 2)], 2)
    full_indices = tf.reshape(full_indices, [-1, 2])

    to_substract = tf.sparse_to_dense(full_indices, x.get_shape(), tf.reshape(values, [-1]), default_value=0.)

    # res should be all 0.
    res = x - to_substract

def stack():
    a = tf.constant([1,2,3])
    b = tf.constant([4,5,6])

    c = tf.stack([a,b],axis=1)
    f = tf.stack([a,b],axis=0)
    d = tf.unstack(c,axis=0)
    e = tf.unstack(c,axis=1)

    print(c.get_shape())
    with tf.Session() as sess:
        print(sess.run(a))
        print(sess.run(b))
        print(sess.run(c))
        print(sess.run(f))

        print(sess.run(d))
        print(sess.run(e))



if __name__ == '__main__':
    # get_top_k()
    stack()
    pass
