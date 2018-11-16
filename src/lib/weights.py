import tensorflow as tf


def gaussian_with_radius(dist, r):
    c = 100
    exp1 = tf.exp(-((dist / c) ** 2))
    exp2 = tf.exp(-((r / c) ** 2))

    return tf.cond(dist <= r, lambda: (exp1 - exp2) / (1 - exp2), lambda: tf.constant(0,dtype=tf.float64), name="inside_gaussian_radius")
