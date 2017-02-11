class Trainer:
	"""
	An internal trainer for the BW colorization model.
	"""
	def __init__(self, sess, input_placeholder, output_tensor, datasource):
		"""
		Initializes the trainer given a collection of model tensors and a dataset.
		"""
		self.model_input = input_placeholder
		self.model_output = output_tensor
		self.datasource = datasource

		self.loss, self.optimizer = self._construct_trainer_graph()

	def _construct_trainer_graph():
		"""
		Constructs the trainer graph for the BWColorization model.
		Returns: A loss tensor and an optimizer.
		"""
		with tf.variable_scope("trainer"):
			pass