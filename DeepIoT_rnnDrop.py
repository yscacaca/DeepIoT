import numpy as np
import tensorflow as tf
import numbers
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import tensor_shape 


def GenRNNMask(keep_prob=1.0, is_training=False, batch_size=64, inter_dim=1, dtype=None, seed=None):
	is_training = ops.convert_to_tensor(is_training, name='is_training')
	noise_shape = (batch_size, inter_dim)
	random_tensor = keep_prob
	random_tensor += random_ops.random_uniform(noise_shape,
												   seed=seed,
												   dtype=tf.float32)
	binary_tensor = math_ops.floor(random_tensor)
	ret = tf.cond(is_training, lambda: tf.identity(binary_tensor), 
									lambda: tf.identity(keep_prob))
	return ret



class DropoutWrapper(tf.contrib.rnn.RNNCell):

	def __init__(self, cell, binary_tensor=None, dtype=None, seed=None):

		self._cell = cell
		self._seed = seed
		self._binary_tensor = binary_tensor


	@property
	def state_size(self):
		return self._cell.state_size

	@property
	def output_size(self):
		return self._cell.output_size

	def zero_state(self, batch_size, dtype):
		with ops.name_scope(type(self).__name__ + "ZeroState", values=[batch_size]):
			return self._cell.zero_state(batch_size, dtype)


	def __call__(self, inputs, state, scope=None):

		output, new_state = self._cell(inputs, state, scope)
		output = output*self._binary_tensor
		new_state = new_state*self._binary_tensor

		return output, new_state


