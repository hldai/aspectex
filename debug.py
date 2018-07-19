# from seqItem import *
# from numpy import *
import string
import tensorflow as tf


def create_model():
    # Create some variables.
    v1 = tf.get_variable("v1", shape=[3], initializer=tf.zeros_initializer)
    # v2 = tf.get_variable("v2", shape=[5], initializer=tf.zeros_initializer)
    v2 = tf.Variable([1, 1, 1, 1, 1], name="v2", dtype=tf.float32)
    v3 = tf.constant([3, 2, 1], dtype=tf.float32, name="v3")

    inc_v1 = v1.assign(v1 + 1)
    dec_v2 = v2.assign(v2 - 1)

    # Add an op to initialize the variables.
    init_op = tf.global_variables_initializer()

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    # Later, launch the model, initialize the variables, do some work, and save the
    # variables to disk.
    with tf.Session() as sess:
        sess.run(init_op)
        # Do some work with the model.
        inc_v1.op.run()
        # dec_v2.op.run()
        # Save the variables to disk.
        save_path = saver.save(sess, "d:/data/tmp/model.ckpt")
        print("Model saved in path: %s" % save_path)


def restore_model():
    v0 = tf.constant([3, 4, 2], dtype=tf.float32)
    v4 = tf.get_variable("v4", shape=[3], initializer=tf.random_normal_initializer)
    v5 = v4 + v0
    g2_init = tf.global_variables_initializer()

    g1 = tf.Graph()
    with g1.as_default():
        # Create some variables.
        v1 = tf.get_variable("v1", shape=[3], initializer=tf.zeros_initializer)
        v2 = tf.get_variable("v2", shape=[5], initializer=tf.zeros_initializer)
        # v2 = tf.Variable([1, 1, 2], name="v2", dtype=tf.float32)
        saver = tf.train.Saver()
        sess = tf.Session()
        saver.restore(sess, "d:/data/tmp/model.ckpt")
        v1_val, v2_val = sess.run([v1, v2])
        print("v1 : %s" % v1_val)
        print("v2 : %s" % v2_val)

    tf.reset_default_graph()

    # Add ops to save and restore all the variables.
    # tf.reset_default_graph()

    # Later, launch the model, use the saver to restore variables from disk, and
    # do some work with the model.
    # with tf.Session(graph=g1) as sess:

    # sess1 = tf.Session(graph=g1)
    sess = tf.Session()

    # sess.run(tf.global_variables_initializer())
    # Restore variables from disk.

    # sess2.run(tf.local_variables_initializer())
    sess.run(g2_init)
    # sess2.run(tf.variables_initializer([v4]))
    v4_val, v5_val = sess.run([v4, v5])
    print(v4_val)
    print(v5_val)

# create_model()
# restore_model()
v = 0.003472
print('{:.4f}'.format(v))

import platform
print(platform.platform())
