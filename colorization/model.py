"""
 model.py
	Author: William Guss

	The main code for the colorization model.
"""

import tensorflow as tf
import numpy as np

import colorization.architecture

class ColorizationModel:
	"""
	The colorization model.
	"""
	def __init__(self, sess, model_path, load=False):
		"""
		Initializes the colorization mode with a session,
		a persistence directory, and a flag for restoration.
		"""
		self.sess = sess

		with tf.variable_scope("colorizer"):
			# Construct the tensorflow computation graph. 
			self.input_placeholder, self.output_tensor = self._construct_model_graph()

			# Build the model saver.		
			self.model_path = model_path
			if not os.path.exists(self.model_path):
				os.mkdir(self.model_path)

			# Grab all of the variables from the scope and construct a tf.train.Saver.
			scope_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=clone_scope.name)
			self.saver = tf.train.Saver(scope_vars)
			if load:
				self.load()
			else:
				# Initialize variables if no saver given.
				init = tf.variables_initializer(scope_vars)
				self.sess.run(init)

	###############################
	#      MODEL ARCHITECTURE     #
	###############################
	def _construct_model_graph(self):
		"""
		Constructs the feedforward graph for colorful image colorization. 
		This is the primary piece of code for the colorization model.
		"""
		input_placeholder = None

		# TODO: Preprocessing
		preprocessed_image = input_placeholder

		ab_color_dist = colorization.architecture.construct(preprocessed_image)


		# TODO: We need also return the conversion version.
		return ab_color_dist


	def feed(self, image):
		"""
		Feeds an image throught the colorization model.
		"""

		colorized_image = self.sess.run(self.output_tensor, {
			self.input_placeholder: [image]
			})[0];

		return colorized_image;


	def get_trainer(self, datasource):
		"""
		Gets a trainer for the model given an datasource abstraction,.
		"""
		return Trainer(
			self.sess,
			self.input_placeholder,
			self.output_tensor,
			datasource)
		

	def save(self):
		self.saver.save(self.sess, self.model_path)

	def load(self):
		self.saver.restore(self.sess, self.model_path)