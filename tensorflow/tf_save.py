
import tensorflow as tf


def save_variable():
    var1 = tf.get_variable(name='var1', shape=[1],
                           initializer=tf.constant_initializer(0))
    var2 = tf.get_variable(name='var2', shape=[1],
                           initializer=tf.constant_initializer(0), trainable=False)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    train_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    global_var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

    def save_variable_var_list(sess, var_list, name):
        saver = tf.train.Saver(var_list=var_list)
        saver.save(sess, name)

    def save_variable_default(sess, name):
        saver = tf.train.Saver()
        saver.save(sess, name)

    def print_variable_list(var_list):
        for var in var_list:
            print(var.name)

    #
    print_variable_list(train_var)
    print_variable_list(global_var)

    save_variable_var_list(sess, train_var, 'save/save_train_var')
    save_variable_default(sess, 'save/save_default_var')

def restore_variable(path_name):
    var1 = tf.get_variable(name='var1', shape=[1],
                           initializer=tf.constant_initializer(0))
    var2 = tf.get_variable(name='var2', shape=[1],
                           initializer=tf.constant_initializer(0), trainable=False)
    sess = tf.Session()
    saver = tf.train.Saver()
    saver.restore(sess, path_name)
    var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    for var in var_list:
        print(var.name)


if __name__ == '__main__':
    # save_variable()
    # restore_variable('save/save_train_var')
    restore_variable('save/save_default_var')