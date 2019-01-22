import sys
sys.path.append('..')
import tensorflow as tf
import numpy as np
from neuralph import utils
from neuralph.neuralph_fitter import NeuralPHFitter as NPH


score = np.array([0.7, 0.5, 0.3, 0.4, 0.2, 0.1, 0.05, 0.2])
# score = score[::-1]
T = np.array([1,2,3,4,5,6,7,8])
E = np.array([0,1,0,0,1,1,1,1])
T_E = np.array(list(zip(T,E)))
print(utils.concordance_index(score, T, E))
print(utils.neg_log_partial_likelihood(score, T, E))

pred_score = tf.constant(score)
T_E = tf.convert_to_tensor(T_E, dtype=np.float32)

c_index = NPH.concordance_index(T_E, pred_score)
l = NPH.neg_log_partial_likelihood(T_E, pred_score)

with tf.Session() as sess:
    # print(sess.run(T_E_shape))
    # print(sess.run(T_shape_after_slice))
    # print(sess.run(E_shape_after_slice))
    # print(sess.run(score_shape_after_slice))
    # print(sess.run(T_shape_after_gather))
    # print(sess.run(E_shape_after_gather))
    # print(sess.run(score_shape_after_gather))
    print(sess.run(c_index))
    print(sess.run(l))

