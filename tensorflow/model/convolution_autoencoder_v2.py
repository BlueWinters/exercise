# tensorflow version 1.10

import tensorflow as tf
import os
import time


class ConvolutionAutoencoder(object):
    def __init__(self, sess, input_shape, encoder_filter, decoder_filter, name='ConvolutionAutoencoder'):
        self.input_shape = input_shape
        self.encoder = encoder_filter
        self.decoder = decoder_filter
        self.name = name
        self.sess = sess

        self.vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

        encoder_filter = [[5, 5, 128], [5, 5, 64], [5, 5, 32]] # [width, height, channel]
        decoder_filter = [[5, 5, 64], [5, 5, 128], [5, 5, 1]]

    def _init_vars(self):
        with tf.variable_scope(self.name) as vs:
            self._init_encoder_vars()
            self._init_decoder_vars()

    def initialize(self, input, is_train):
        n_encoder = len(self.encoder)
        channels_set = [self.input_shape[-1]] # input channels
        for n in range(n_encoder):
            channels_set.append(self.encoder[n][-1])
        for n in range(n_encoder):
            self._set_layer_filter_vars(width=self.encoder[n][0], height=self.encoder[n][0],
                                        in_chls=channels_set[n], out_chls=channels_set[n+1],
                                        name='layer_'+str(n))
        n_decoder = len(self.decoder)
        channels_set = [self.encoder[-1][-1]] # middle channels
        for n in range(n_decoder):
            channels_set.append(self.decoder[n][-1])

        for n in range(n_decoder):
            self._set_layer_filter_vars(width=self.decoder[n][0], height=self.decoder[n][0],
                                        in_chls=channels_set[n], out_chls=channels_set[n+1],
                                        name='layer_'+str(n))

    def _set_layer_filter_vars(self, width, height, in_chls, out_chls, name, std=1.):
        with tf.variable_scope(name) as vs:
            k = tf.get_variable('filter', [width, height, in_chls, out_chls],
                                initializer=tf.truncated_normal_initializer(stddev=std))
            b = tf.get_variable('biases', [out_chls],
                                initializer=tf.constant_initializer(0))
        return k, b

    def _conv(self, input, name):
        with tf.variable_scope(name, reuse=True) as vs:
            W = tf.get_variable('filter')
            b = tf.get_variable('biases')
            conv = tf.nn.conv2d(input, W, [1, 1, 1, 1], "SAME") + b
        return conv

    def loss(self, input):
        n_encoder = len(self.encoder)
        n_decoder = len(self.decoder)
        h = []
        h.append(input)

        with tf.variable_scope(self.name) as vs:
            # encoder
            with tf.variable_scope('encoder', reuse=True) as vs:
                for n, in range(n_encoder):
                    h.append(self._conv(h[-1], 'layer_'+str(n)))
            # decoder
            with tf.variable_scope('decoder', reuse=True) as vs:
                for n in range(n_decoder):
                    h.append(self._conv(h[-1], 'layer_'+str(n)))
        # return MSE loss
        return tf.reduce_mean(tf.square(h[-1] - input))

    def save(self, path):
        saver = tf.train.Saver(self.vars)
        saver.save(self.sess, path)

    def restore(self, path):
        saver = tf.train.Saver(self.vars)
        saver.restore(self.sess, path)

    def visual_filter(self):
        pass


if __name__ == '__main__':
    # set parameters
    encoder_filter = [[5, 5, 128], [5, 5, 64], [5, 5, 32]] # [width, height, channel]
    decoder_filter = [[5, 5, 64], [5, 5, 128], [5, 5, 1]]
    num_epochs = 100
    batch_size = 100
    learn_rate = 1e-3
    input_shape = [28, 28, 1]
    shape = [batch_size, 28, 28, 1]