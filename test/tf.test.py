import tensorflow as tf
from neuralph.utils import _valid_comparison, _concordance_value
import numpy as np

pred_score = tf.constant(np.array([0.7, 0.5, 0.3, 0.4, 0.2, 0.1, 0.05, 0.2]))
T = tf.constant(np.array([1,2,3,4,5,6,7,8]))
E = tf.constant(np.array([0,1,0,0,1,1,1,1]))

def func(x):
    a, b = x[0], x[1]
    s = 0
    for i in range(len(a)):
        s += a[i] + b[i]
    return s
        



score = tf.constant(np.array([0.7, 0.5, 0.3, 0.4, 0.2, 0.1, 0.05, 0.2]))
score_mat = tf.subtract(score, tf.reshape(score, [-1,1]))

T = tf.constant(np.array([1,2,3,4,5,6,7,8]))
T_mat = tf.subtract(T, tf.reshape(T, [-1,1]))

E = tf.constant(np.array([0,1,0,0,1,1,1,1]))
E_mat_either = tf.add(E, tf.reshape(E, [-1,1]))
E_mat_diff = tf.subtract(E, tf.reshape(E, [-1,1]))

valid_comp1 = tf.logical_and(tf.equal(T_mat, 0), tf.equal(E_mat_either, 1))
valid_comp2 = tf.logical_and(tf.not_equal(T_mat, 0), tf.equal(E_mat_either, 2))
valid_comp3 = tf.logical_and(tf.equal(E_mat_diff, -1), tf.greater(T_mat, 0))
valid_comp = tf.logical_or(tf.logical_or(valid_comp1, valid_comp2), valid_comp3)
valid_comp = tf.linalg.band_part(valid_comp, 0, -1)

random = tf.logical_and(tf.equal(score_mat, 0), valid_comp)
good1 = \
tf.logical_and(
    tf.logical_and(
        tf.greater(score_mat, 0), tf.greater(T_mat, 0)
    ), valid_comp
)
good2 = \
tf.logical_and(
    tf.logical_and(
        tf.logical_and(
            tf.greater(score_mat, 0), tf.equal(T_mat, 0)
        ), tf.equal(E_mat_diff, -1)
    ), valid_comp
)

random = tf.multiply(tf.reduce_sum(tf.to_float(random)), 0.5)
good = tf.add(
    tf.reduce_sum(tf.to_float(good1)), 
    tf.reduce_sum(tf.to_float(good2))
)
pairs = tf.reduce_sum(tf.to_float(valid_comp))
c_index = tf.divide(tf.add(random, good), pairs)


with tf.Session() as sess:
    print(sess.run(c_index))

