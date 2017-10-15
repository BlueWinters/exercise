
import tensorflow as tf

layer1 = {}
layer2 = {}


def var_initializer(layer_set):
    pass

def init_classifier(in_dim, h_dim, out_dim):
    layer1['w'] = tf.get_variable(name='layer1_w', shape=[None,in_dim], dtype=tf.float32,
                                  initializer=tf.random_normal())
    layer1['b'] = 0