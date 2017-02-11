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
		   pad=0, stride=1, activation=tf.nn.relu, name="conv"): 
	"""
	Builds a convolution layer.
	"""
	with tf.variable_scope(name):
		weights = uniform_random_tensor(
			[kernel_size, kernel_size, 
			input_channels, output_channels])
		stride = [1, stride, stride, 1]
		pad_vec =  [[0,0], [pad, pad], [pad, pad], [0,0]]

		padded_input = tf.pad(input_tensor, pad_vec, "CONSTANT")
		# Create the convolution using valid padding.
		# Valid padding basically does no padding and relies on 
		# tf.pad to specify the padding.
		convolution = tf.nn.conv2d(padded_input, weights, stride, padding="VALID")

		# TODO: Add a bias vector:

		output = activation(convolution)

	return convolution

