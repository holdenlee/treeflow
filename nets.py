import tensorflow as tf
from functools import *
import itertools
import inspect
from datetime import datetime
import re
import os.path
import time
import numpy as np
from six.moves import xrange
from utils import *
from random import *

def add_bias(x,b,name=None):
    return tf.add(x,b,name)

def linear_layer(x,W,b, name=None):
    return tf.add(tf.matmul(x, W), b, name)

#softmax over batches, l is the number in the batch. (For a single one, l=1.)
def softmax_layer(x,W,b, name=None):
    """R^{m*n}, R^n, R^{l*m} -> R^{l*n}"""
    return tf.nn.softmax(tf.matmul(x,W) + b, name)
#Alternatively use to add b:
#tf.nn.bias_add(value, bias, data_format=None, name=None)

def relu_layer(x,W,b,name=None):
    """R^{m*n}, R^n, R^{l*m} -> R^{l*n}"""
    return tf.nn.relu(tf.matmul(x,W) + b, name)

#y is actual, yhat is predicted
def cross_entropy(y, yhat, t=0):
    """R^{l*n}, R^{l*n} -> R"""
    return tf.reduce_mean(-tf.reduce_sum(y * logt(yhat,t), reduction_indices=[-1]))
    #-sum(y *. log(yhat)) where sum is along first axis
    #now take mean

def correct_prediction(y, yhat):
    """R^{l*n}, R^{l*n} -> Bool^l"""
    return tf.equal(tf.argmax(yhat,1), tf.argmax(y,1))

def accuracy(y, yhat):
    """R^{l*n}, R^{l*n} -> R^l"""
    return tf.reduce_mean(tf.cast(correct_prediction(y,yhat), tf.float32))

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool(x, r, c):
  return tf.nn.max_pool(x, ksize=[1, r, c, 1],
                        strides=[1, r, c, 1], padding='SAME')

# If a model is trained with multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'
def activation_summary(x):
  """Helper to create summaries for activations.

  Creates a summary that provides a histogram of activations.
  Creates a summary that measure the sparsity of activations.

  Args:
    x: Tensor
  Returns:
    x: Tensor (same thing)
  """
  # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
  # session. This helps the clarity of presentation on tensorboard.
  tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
  tf.histogram_summary(tensor_name + '/activations', x)
  tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))
  return x

"""
def fold(f, args, x):
    return reduce(lambda y, args: f(y,**args), x)
"""

def weight_decay(var,wd,add=True):
    loss = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
    if add:
        tf.add_to_collection('losses', loss)
    return loss

#join = itertools.chain.from_iterable

def eval_with_missing_args(f,x,d,var_dict):
    t = inspect.getargspec(f)
    vars_ = t[0]
    specs = t[1:]
    #print("attempt eval")
    #print(vars_)
    #print(t)
    if t[-1]==None:
        l = 0
    else:
        l=len(t[-1])
    reqs = vars_[1:len(vars_)-l]
    #tf.get_variable_scope().reuse_variables()
    #http://stackoverflow.com/questions/6486450/python-compute-list-difference
    sc = tf.get_variable_scope().name
    d2=dict([(y, var_dict[sc+"/"+y]) for y in reqs if y not in d.keys()])
    #variable_on_cpu(y)
    d2.update(d)
    return f(x,**d2)

def add_loss_summaries(total_loss, losses=[], decay=0.9):
  """Add summaries for losses in model.

  Generates moving average for all losses and associated summaries for
  visualizing the performance of the network.

  Args:
    total_loss: Total loss from loss().
  Returns:
    loss_averages_op: op for generating moving averages of losses.
  """
  # Compute the moving average of all individual losses and the total loss.
  # shadow_variable = decay * shadow_variable + (1 - decay) * variable
  loss_averages = tf.train.ExponentialMovingAverage(decay, name='avg')
  # losses = tf.get_collection('losses')
  # Instead pass in `losses` as an argument.
  loss_averages_op = loss_averages.apply(losses + [total_loss])

  # Attach a scalar summary to all individual losses and the total loss; do the
  # same for the averaged version of the losses.
  for l in losses + [total_loss]:
    # Name each loss as '(raw)' and name the moving average version of the loss
    # as the original loss name.
    tf.scalar_summary(l.op.name +' (raw)', l)
    tf.scalar_summary(l.op.name, loss_averages.average(l))
  return loss_averages_op

def train_step(total_loss, losses, global_step, optimizer, 
               summary_f=None):
  """Train model.

  Create an optimizer and apply to all trainable variables. Add moving
  average for all trainable variables.

  Args:
    total_loss: Total loss from loss().
    global_step: Integer Variable counting the number of training steps
      processed.
  Returns:
    train_op: op for training.
  """
  # decayed_learning_rate = learning_rate *
  #                     decay_rate ^ (global_step / decay_steps)
  # Variables that affect learning rate.
  
  # Generate moving averages of all losses and associated summaries.
  loss_averages_op = add_loss_summaries(total_loss, losses)

  # Compute gradients.
  with tf.control_dependencies([loss_averages_op]):
    #must compute loss_averages_op before executing this---Why?
    opt = optimizer(global_step)
    grads = opt.compute_gradients(total_loss)

  # Apply gradients.
  apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

  # Add histograms for trainable variables.
  for var in tf.trainable_variables():
    tf.histogram_summary(var.op.name, var)

  # Add histograms for gradients.
  for grad, var in grads:
    if grad is not None:
      tf.histogram_summary(var.op.name + '/gradients', grad)

  deps = [apply_gradient_op]

  if summary_f!=None:
      # Track the moving averages of all trainable variables.
      variable_averages = summary_f(global_step)
      variables_averages_op = variable_averages.apply(tf.trainable_variables())
      deps.append(variables_averages_op)

  with tf.control_dependencies(deps):
    train_op = tf.no_op(name='train')

  return train_op

def valid_pos_int(n):
    return n!=None and n>0

def train2(funcs, step_f, output_steps=10, summary_steps=100, save_steps=1000, eval_steps = 1000, max_steps=1000000, train_dir="/", log_device_placement=False, batch_size=128,train_data=None,validation_data=None, test_data=None, train_feed={}, eval_feed={}, args_pl=None, batch_feeder_args=[], fn=None, verbosity=1):
    global_step = tf.Variable(0, trainable=False)
    loss = funcs["loss"]
    # Build a Graph that trains the model with one batch of examples and
    # updates the model parameters.
    train_op = step_f(funcs, global_step)

    # Create a saver.
    saver = tf.train.Saver(tf.all_variables())

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.merge_all_summaries()

    # Build an initialization operation to run below.
    init = tf.initialize_all_variables()

    # Start running operations on the Graph.
    sess = tf.Session() #config=tf.ConfigProto(
        #log_device_placement))
    sess.run(init)

    # Start the queue runners.
    tf.train.start_queue_runners(sess=sess)

    summary_writer = tf.train.SummaryWriter(train_dir, sess.graph)
    printv("Initialized.", verbosity, 1)
    for step in xrange(max_steps):
      start_time = time.time()
      feed_dict = map_feed_dict(merge_two_dicts(
          fill_feed_dict(train_data, batch_size, 
                         args_pl), train_feed))
#                     placeholder_dict["y_"]),
#      {"fc/keep_prob:0": 0.5})
#      if feed_dict_fun!=None:
#          feed_dict = map_feed_dict(feed_dict_fun())
      #print feed_dict
      _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)
      duration = time.time() - start_time

      assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
      if valid_pos_int(output_steps) and step % output_steps == 0:
         output_info(batch_size, step, loss_value, duration)
      if valid_pos_int(summary_steps) and step % summary_steps == 0:
        printv("Running summary...", verbosity, 1)
        summary_str = sess.run(summary_op, feed_dict=feed_dict)
        summary_writer.add_summary(summary_str, step)
      # Save the model checkpoint periodically.
      if (valid_pos_int(save_steps) and step % save_steps == 0) or (step + 1) == max_steps:
        checkpoint_path = os.path.join(train_dir, 'model.ckpt')
        printv("Saving as %s" % checkpoint_path, verbosity, 1)
        saver.save(sess, checkpoint_path, global_step=step)
      if ((valid_pos_int(eval_steps) and (step + 1) % (eval_steps) == 0)) or (eval_steps!=None and (step + 1) == max_steps):
        for (data, name) in zip([train_data,validation_data,test_data], ["Training", "Validation", "Test"]):
            if data!=None:
                printv('%s Data Eval:' % name, verbosity, 1)
                do_eval(sess,
                        funcs["accuracy"],
                        data,
                        batch_size,
                        args_pl,
                        batch_feeder_args,
                        eval_feed)
      if fn != None:
        fn(sess)
#    li = tf.get_collection(tf.GraphKeys.VARIABLES)
#    for i in li:
#        print(i.name)
#    print(tf.get_default_graph().get_tensor_by_name("layer1//W:0"))
    return sess
    

def train(fs, step_f, output_steps=10, summary_steps=100, save_steps=1000, eval_steps = 1000, max_steps=1000000, train_dir="/", log_device_placement=False, batch_size=128,train_data=None,validation_data=None, test_data=None, train_feed={}, eval_feed={}, args_pl=None, batch_feeder_args=[] ,verbosity=1):
    """
    Train model.
  
    Args:
      fs: Inference function and loss function.
      step: Function to execute at each training step,
        takes arguments `loss` and `global_step`
    Returns:
      None.
    """
    #global counter
#   with tf.Graph().as_default():
    printv("Building graph...",verbosity,1)
    funcs = fs()
    return train2(funcs, step_f, output_steps, summary_steps, save_steps, eval_steps, max_steps, train_dir, log_device_placement, batch_size,train_data,validation_data, test_data, train_feed, eval_feed, args_pl, batch_feeder_args)

def _conv2d_dims(inp, kern, filt, padding='SAME'):
    if padding == 'SAME':
        return ceil(float(inp)/float(strides))
    else:
        return ceiling(float(inp - kern + 1)/float(strides))

def get_conv2d_dims(input_dim, filter_dim, strides, padding='SAME'):
    in_height, in_width, in_channels = input_dim
    filter_height, filter_width, _, out_channels = filter_dim
    _, stride_height, stride_width, _ = strides
    return [_conv2d_dims(in_height, filter_height, stride_height),
            _conv2d_dims(in_width, filter_width, stride_width),
            out_channels]

def variable_on_cpu(name, shape=None, initializer=None,dtype=tf.float32, var_type="variable"):
  """Helper to create a Variable stored on CPU memory.

  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable

  Returns:
    Variable Tensor
  """
  with tf.device('/cpu:0'):
      if var_type == "variable":
          var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
      else: #"placeholder"
          #print(dtype, shape, name)
          var = tf.placeholder(dtype, shape=shape, name=name)
  return var

def fill_feed_dict(batch_feeder, batch_size, args_pl=None, args = []):
  """Fills the feed_dict for training the given step. Args should be a list.
  A feed_dict takes the form of:
  feed_dict = {
      <placeholder>: <tensor of values to be passed for placeholder>,
      ....
  }"""
  # Create the feed_dict for the placeholders filled with the next
  # `batch size ` examples.
  b = batch_feeder.next_batch(batch_size, *args)
  if args_pl != None:
      return {args_pl[k] : b[k] for (k,v) in args_pl.items()}
  else:
      return b

class BatchFeeder(object):

  def __init__(self, args, num_examples, next_batch_fun):
      self.args = args
      self.index = 0
      if num_examples != None:
          self.num_examples = num_examples
      else:
          self.num_examples = len(args.values()[0])
      self.epochs_completed = 0
      self.next_batch_fun = next_batch_fun

  def next_batch(self, batch_size, *args):
      b = self.next_batch_fun(self, batch_size, *args)
      # print(b)
      return b

def map_feed_dict(feed_dict):
    #print(feed_dict)
    return map_keys(lambda x: tf.get_default_graph().get_tensor_by_name(x) if isinstance(x,str) else x, feed_dict)
#http://stackoverflow.com/questions/394809/does-python-have-a-ternary-conditional-operator
#http://stackoverflow.com/questions/644178/how-do-i-re-map-python-dict-keys

def shuffle_refresh(bf):
    perm = np.arange(bf.num_examples)
    np.random.shuffle(perm)
    bf.args = {k : v[perm] for (k,v) in bf.args.items()}

def batch_feeder_f(bf, batch_size, refresh_f = shuffle_refresh):
    if bf.index == 0 and bf.epochs_completed == 0:
        refresh_f(bf)
    start = bf.index
    bf.index += batch_size
    # for simplicity, discard those at end.
    if bf.index > bf.num_examples:
      # Finished epoch
      bf.epochs_completed += 1
      # Shuffle the data
      refresh_f(bf)
      # Start next epoch
      start = 0
      bf.index = batch_size
      #print(batch_size)
      #print(bf.num_examples)
      assert batch_size <= bf.num_examples
    end = bf.index
    return { k : v[start:end] for (k,v) in bf.args.items()}

def make_batch_feeder(args, refresh_f=shuffle_refresh, num_examples = None):
    if num_examples==None:
        l = len(args.values()[0])
    else:
        l = num_examples
    return BatchFeeder(args, l, (lambda bf, batch_size: batch_feeder_f(bf, batch_size, refresh_f)))

def do_eval(sess,
            eval_correct,
            batch_feeder, 
            batch_size,
            args_pl={},
            args=[],
            eval_feed={},
            verbosity=1):
  """Runs one evaluation against the full epoch of data.
  Args:
    sess: The session in which the model has been trained.
    eval_correct: The Tensor that returns the number of correct predictions.
    xs_placeholder: The images placeholder.
    ys_placeholder: The labels placeholder.
    batch_feeder: The set of xs and ys to evaluate.
  """
  # And run one epoch of eval.
  true_count = 0  # Counts the number of correct predictions.
  steps_per_epoch = batch_feeder.num_examples // batch_size
  num_examples = steps_per_epoch * batch_size
  for step in xrange(steps_per_epoch):
    feed_dict = fill_feed_dict(batch_feeder, batch_size, args_pl, args = args)
    #?
    feed_dict.update(map_feed_dict(eval_feed))
    true_count += sess.run(eval_correct, feed_dict=feed_dict)
  precision = float(true_count) / float(num_examples)
  printv('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %
         (num_examples, true_count, precision), verbosity, 1)

"""
based off code at https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/learn/python/learn/datasets/mnist.py
"""

#c1, c2 are two collections of variables.
def am_train_step(total_loss, losses, global_step, optimizer, c1, c2,
               summary_f=None):
    loss_averages_op = add_loss_summaries(total_loss, losses)
    # Compute gradients.
    with tf.control_dependencies([loss_averages_op]):
        #must compute loss_averages_op before executing this---Why?
        opt = optimizer(global_step)
        grads = opt.compute_gradients(total_loss, c1 if global_step % 2 == 0 else c2)

    # Apply gradients.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # Add histograms for trainable variables.
    hist_trainables()

    # Add histograms for gradients.
    hist_grads(grads)

    deps = [apply_gradient_op]
  
    apply_summary_f(summary_f, global_step, deps)

    #lump them together
    return get_train_op(deps)


def reuse_var(scope, var):
    with tf.variable_scope(scope, reuse=True):
    #scope.reuse_variables()
        return tf.get_variable(var)

      
def output_info(batch_size, step, loss_value, duration):
    examples_per_sec = batch_size / duration
    sec_per_batch = float(duration)
    format_str = ('%s: step %d, loss = %.5f (%.1f examples/sec; %.3f '
                  'sec/batch)')
    print (format_str % (datetime.now(), step, loss_value,
                         examples_per_sec, sec_per_batch))

def is_gtr(x, c):
    return (tf.sign(x-c) + 1)/2

def is_less(x,c):
    return (1-tf.sign(x-c))/2

def logt(x, t=0):
    return tf.log(tf.maximum(x, t))

def select(ps):
    s = 0
    r = random()
    for (i,p) in enumerate(ps):
        s = s+p
        if r < s:
            return i
    return None
