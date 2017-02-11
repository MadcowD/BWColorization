"""
 architecture.pu
 	Author: William H. Guss

 	A single file for the main architecture of the colorization model.
"""


import tensorflow as tf
import numpy as np

from colorization.utils import conv2d

def construct(input_placeholder):
	"""
	Constructs the main model architecture in tensorflow.
	"""
		###############################
		#      MODEL ARCHITECTURE     #
		###############################
		# First block of convolutions
		with tf.variable_scope("conv_block_1"):
			conv_1_1 = conv2d(input_placeholder,
				input_channels=1,
				output_channels=64,
				kernel_size=3,
				pad=1)
			conv_1_2 = conv2d(conv_1_1,
				input_channels=64,
				output_channels=64,
				kernel_size=3,
				pad=1,
				stride=2)
			# TODO batchn
			bn_1 = conv_1_2

		# Second block of convolutions.
		with tf.variable_scope("conv2"):
			conv_2_1 = conv2d(bn_1,
				input_channels=64,
				output_channels=128,
				kernel_size=3,
				pad=1)
			conv_2_2 = conv2d(conv_2_1,
				input_channels=128,
				output_channels=128,
				kernel_size=3,
				pad=1,
				stride=2)

			# TODO batchn
			bn_2 = conv_2_2

		with tf.variable_scope("conv3"):
			conv_3_1 = conv2d(bn_2,
				input_channels=128,
				output_channels=256,
				kernel_size=3,
				pad=1)
			conv_3_2 = conv2d(conv_3_1,
				input_channels=256,
				output_channels=256,
				kernel_size=3,
				pad=1)
			conv_3_3 = conv2d(conv_3_2,
				input_channels=256,
				output_channels=256,
				kernel_size=3,
				pad=1,
				stride=2)
			# TODO batchn
			bn_3 = conv_3_3


		with tf.variable_scope("conv4"):
			conv_3_1 = conv2d(bn_2,
				input_channels=128,
				output_channels=256,
				kernel_size=3,
				pad=1)
			conv_3_2 = conv2d(conv_3_1,
				input_channels=256,
				output_channels=256,
				kernel_size=3,
				pad=1)
			conv_3_3 = conv2d(conv_3_2,
				input_channels=256,
				output_channels=256,
				kernel_size=3,
				pad=1,
				stride=2)
			# TODO batchn
			bn_3 = conv_3_3

		# DILATED LAYERS:
		with tf.variable_scope("conv5"):
			conv_3_1 = conv2d(bn_2,
				input_channels=128,
				output_channels=256,
				kernel_size=3,
				pad=1)
			conv_3_2 = conv2d(conv_3_1,
				input_channels=256,
				output_channels=256,
				kernel_size=3,
				pad=1)
			conv_3_3 = conv2d(conv_3_2,
				input_channels=256,
				output_channels=256,
				kernel_size=3,
				pad=1,
				stride=2)
			# TODO batchn
			bn_3 = conv_3_3

		with tf.variable_scope("conv6"):
			conv_3_1 = conv2d(bn_2,
				input_channels=128,
				output_channels=256,
				kernel_size=3,
				pad=1)
			conv_3_2 = conv2d(conv_3_1,
				input_channels=256,
				output_channels=256,
				kernel_size=3,
				pad=1)
			conv_3_3 = conv2d(conv_3_2,
				input_channels=256,
				output_channels=256,
				kernel_size=3,
				pad=1,
				stride=2)
			# TODO batchn
			bn_3 = conv_3_3


		with tf.variable_scope("conv7"):
			conv_3_1 = conv2d(bn_2,
				input_channels=128,
				output_channels=256,
				kernel_size=3,
				pad=1)
			conv_3_2 = conv2d(conv_3_1,
				input_channels=256,
				output_channels=256,
				kernel_size=3,
				pad=1)
			conv_3_3 = conv2d(conv_3_2,
				input_channels=256,
				output_channels=256,
				kernel_size=3,
				pad=1,
				stride=2)
			# TODO batchn
			bn_3 = conv_3_3

		with tf.variable_scope("conv8"):
			conv_3_1 = conv2d(bn_2,
				input_channels=128,
				output_channels=256,
				kernel_size=3,
				pad=1)
			conv_3_2 = conv2d(conv_3_1,
				input_channels=256,
				output_channels=256,
				kernel_size=3,
				pad=1)
			conv_3_3 = conv2d(conv_3_2,
				input_channels=256,
				output_channels=256,
				kernel_size=3,
				pad=1,
				stride=2)
			# TODO batchn
			bn_3 = conv_3_3
