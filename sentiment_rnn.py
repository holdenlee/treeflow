import tensorflow as tf
import numpy as np
from treeflow import treeflow, treeflow_unfold

#treeflow(fs, child_inds, leaves_inds, leaves, node_inputs = [], has_outputs = False, is_list_fn = False, def_size=10, ans_type=tf.float32, output_type=tf.float32, degree=2)

"""
* `T` is tensor
"""
def rtn(T, b, f, child_inds, leaves_inds, leaves, *node_inputs):
    def tcombine(x1, x2, *inps):
        #https://www.tensorflow.org/api_docs/python/math_ops/matrix_math_functions#matmul
        y = tf.tanh(tf.add(tf.matmul(tf.matmul(tf.transpose(x1), T), x2), b))
        o = f(y, *inp)
        return (y, o)
    return treeflow(tcombine, child_inds, leaves_inds, leaves, *node_inputs)

