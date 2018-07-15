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
    tf.reset_default_graph()

    # Create some variables.
    v1 = tf.get_variable("v1", shape=[3], initializer=tf.zeros_initializer)
    # v2 = tf.get_variable("v2", shape=[5], initializer=tf.zeros_initializer)
    v2 = tf.Variable([1, 1, 2, 1, 1], name="v2", dtype=tf.float32)
    v3 = tf.constant([3, 2, 2], dtype=tf.float32, name="v3")

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    # Later, launch the model, use the saver to restore variables from disk, and
    # do some work with the model.
    with tf.Session() as sess:
        # sess.run(tf.global_variables_initializer())
        # Restore variables from disk.
        saver.restore(sess, "d:/data/tmp/model.ckpt")
        print("Model restored.")
        # Check the values of the variables
        print("v1 : %s" % v1.eval())
        print("v2 : %s" % v2.eval())
        print("v3 : %s" % v3.eval())


create_model()
restore_model()
