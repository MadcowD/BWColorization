"""
 architecture.pu
 	Author: William H. Guss

 	A single file for the main architecture of the colorization model.
"""


import tensorflow as tf
import numpy as np

from .utils import conv2d

def construct(input_placeholder):
	"""
	Constructs the main model architecture in tensorflow.
	"""
		###############################
		#      MODEL ARCHITECTURE     #
		###############################
		# First block of convolutions
		with tf.variable_scope("conv_1"):
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


		# DILATED LAYERS:
		with tf.variable_scope("conv4"):
			conv_4_1 = conv2d(bn_3,
				input_channels=256,
				output_channels=512,
				kernel_size=3,
				pad=1,
				dilation=1)
			conv_4_2 = conv2d(conv_4_1,
				input_channels=512,
				output_channels=512,
				kernel_size=3,
				pad=1,
				dilation=1)
			conv_4_3 = conv2d(conv_4_2,
				input_channels=512,
				output_channels=512,
				kernel_size=3,
				pad=1,
				dilation=1)
			# TODO batchn
			bn_4 = conv_4_3

		with tf.variable_scope("conv5"):
			conv_5_1 = conv2d(bn_4,
				input_channels=512,
				output_channels=512,
				kernel_size=3,
				pad=2,
				dilation=2)
			conv_5_2 = conv2d(conv_5_1,
				input_channels=512,
				output_channels=512,
				kernel_size=3,
				pad=2,
				dilation=2)
			conv_5_3 = conv2d(conv_5_2,
				input_channels=512,
				output_channels=512,
				kernel_size=3,
				pad=2,
				dilation=2)
			# TODO batchn
			bn_5 = conv_5_3

		with tf.variable_scope("conv6"):
			conv_6_1 = conv2d(bn_5,
				input_channels=512,
				output_channels=512,
				kernel_size=3,
				pad=2,
				dilation=2)
			conv_6_2 = conv2d(conv_6_1,
				input_channels=512,
				output_channels=512,
				kernel_size=3,
				pad=2,
				dilation=2)
			conv_6_3 = conv2d(conv_6_2,
				input_channels=512,
				output_channels=512,
				kernel_size=3,
				pad=2,
				dilation=2)
			# TODO batchn
			bn_6 = conv_6_3


		with tf.variable_scope("conv7"):
			conv_7_1 = conv2d(bn_6,
				input_channels=512,
				output_channels=512,
				kernel_size=3,
				pad=1,
				dilation=1)
			conv_7_2 = conv2d(conv_7_1,
				input_channels=512,
				output_channels=512,
				kernel_size=3,
				pad=1,
				dilation=1)
			conv_7_3 = conv2d(conv_7_2,
				input_channels=512,
				output_channels=512,
				kernel_size=3,
				pad=1,
				dilation=1)
			# TODO batchn
			bn_7 = conv_7_3


		with tf.variable_scope("conv8"):
			conv_8_1 = deconv2d(bn_7,
				input_channels=512,
				output_size=[None, 64, 64, 256],
				kernel_size=4,
				stride=2,
				pad=1)
			conv_8_2 = conv2d(conv_8_1,
				input_channels=256,
				output_channels=256,
				kernel_size=3,
				pad=1)
			conv_8_3 = conv2d(conv_8_2,
				input_channels=256,
				output_channels=256,
				kernel_size=3,
				pad=1,
				stride=1)
			conv_8_313 = conv2d(conv_8_3,
				input_channels=256,
				output_channels=313,
				kernel_size=3,
				pad=1,
				stride=1)


		return conv_8_313
