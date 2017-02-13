"""
 utils.py
	Author: William Guss

	Basic building blocks for tensorflow.
"""

import tensorflow as tf
import numpy as np

def uniform_random_tensor(shape, name="weights"):
	"""
	Makes a tensor of shape SHAPE with uniform random intiialization.
	"""
	with tf.variable_scope(name):
		tensor = tf.Variable(tf.random_uniform(shape, -1, 1))
	return tensor


def conv2d(input_tensor, 
       input_channels, output_channels, kernel_size,  
       pad=0, stride=1, dialation=1, activation=tf.nn.relu, name="conv", custom_weights=None): 
  """
  Builds a convolution layer.
  """
  with tf.variable_scope(name):
    if custom_weights:
      weights = tf.Variable(custom_weights=None)
    else:
      weights = uniform_random_tensor(
        [kernel_size, kernel_size, 
        input_channels, output_channels])
    stride = [1, stride, stride, 1]
    pad_vec =  [[0,0], [pad, pad], [pad, pad], [0,0]]

    padded_input = tf.pad(input_tensor, pad_vec, "CONSTANT")
    # Create the convolution using valid padding.
    # Valid padding basically does no padding and relies on 
    # tf.pad to specify the padding.
    if dilation == 1:
      convolution = tf.nn.conv2d(padded_input, weights, stride, padding="VALID")
    else:
      convolution = tf.nn.atrous_conv2d(padded_input, weights,  dialation, padding="VALID")


    # TODO: Add a bias vector:

    output = activation(convolution)

  return convolution


def deconv2d(input_tensor, 
       input_channels, output_shape, kernel_size,  
       pad=0, stride=1, activation=tf.nn.relu, name="deconv"): 
  """
  Builds a convolution layer.
  """
  with tf.variable_scope(name):
    weights = uniform_random_tensor(
      [kernel_size, kernel_size, 
      output_shape[-1], input_channels])
    pad_vec =  [[0,0], [pad, pad], [pad, pad], [0,0]]
    stride = [1, stride, stride, 1]

    padded_input = tf.pad(input_tensor, pad_vec, "CONSTANT")
    # Create the convolution using valid padding.
    # Valid padding basically does no padding and relies on 
    # tf.pad to specify the padding.
    deconvolution = tf.nn.conv2d_transpose(padded_input, weights, output_shape, stride, padding="VALID")


    # TODO: Add a bias vector:

    output = activation(convolution)

  return convolution

