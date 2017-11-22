import numpy as np
import tensorflow as tf
import numbers
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import tensor_shape 

def dropout(x, keep_prob, is_training=False, noise_shape=None, seed=None, name=None):  
	with ops.name_scope(name, "dropout", [x]) as name:
		x = ops.convert_to_tensor(x, name="x")
		is_training = ops.convert_to_tensor(is_training, name='is_training')
		keep_prob = ops.convert_to_tensor(keep_prob,
										dtype=x.dtype,
										name="keep_prob")

		noise_shape = noise_shape if noise_shape is not None else array_ops.shape(x)
		random_tensor = keep_prob
		random_tensor += random_ops.random_uniform(noise_shape,
												   seed=seed,
												   dtype=x.dtype)
		binary_tensor = math_ops.floor(random_tensor)
		ret = tf.cond(is_training, lambda: x * binary_tensor, 
									lambda: x * keep_prob)

		ret.set_shape(x.get_shape())
		return ret, binary_tensor
		
