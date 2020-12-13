""" Wrapper functions for TensorFlow layers.

Author: Charles R. Qi
Date: November 2016

Upadted by Yue Wang and Yongbin Sun

Further improved by Liang PAN
"""

import numpy as np
import tensorflow as tf
import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/sampling'))
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/grouping'))
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/3d_interpolation'))
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/pointSIFT_op'))

from tf_sampling import farthest_point_sample, gather_point
from tf_grouping import query_ball_point, group_point, knn_point, select_top_k
from tf_interpolate import three_nn, three_interpolate
import tensorflow as tf






def _variable_on_cpu(name, shape, initializer, use_fp16=False, trainable=True):
	"""Helper to create a Variable stored on CPU memory.
	Args:
		name: name of the variable
		shape: list of ints
		initializer: initializer for Variable
	Returns:
		Variable Tensor
	"""
	with tf.device('/cpu:0'):
		dtype = tf.float16 if use_fp16 else tf.float32
		var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype, trainable=trainable)
	return var

def _variable_with_weight_decay(name, shape, stddev, wd, use_xavier=True):
	"""Helper to create an initialized Variable with weight decay.

	Note that the Variable is initialized with a truncated normal distribution.
	A weight decay is added only if one is specified.

	Args:
		name: name of the variable
		shape: list of ints
		stddev: standard deviation of a truncated Gaussian
		wd: add L2Loss weight decay multiplied by this float. If None, weight
				decay is not added for this Variable.
		use_xavier: bool, whether to use xavier initializer

	Returns:
		Variable Tensor
	"""
	if use_xavier:
		initializer = tf.contrib.layers.xavier_initializer()
	else:
		initializer = tf.truncated_normal_initializer(stddev=stddev)
	var = _variable_on_cpu(name, shape, initializer)
	if wd is not None:
		weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
		tf.add_to_collection('losses', weight_decay)
	return var


def conv1d(inputs,
			num_output_channels,
			kernel_size,
			scope,
			stride=1,
			padding='SAME',
			use_xavier=True,
			stddev=1e-3,
			weight_decay=0.0,
			activation_fn=tf.nn.relu,
			bn=False,
			bn_decay=None,
			is_training=None,
			is_dist=False):
	""" 1D convolution with non-linear operation.

	Args:
		inputs: 3-D tensor variable BxLxC
		num_output_channels: int
		kernel_size: int
		scope: string
		stride: int
		padding: 'SAME' or 'VALID'
		use_xavier: bool, use xavier_initializer if true
		stddev: float, stddev for truncated_normal init
		weight_decay: float
		activation_fn: function
		bn: bool, whether to use batch norm
		bn_decay: float or float tensor variable in [0,1]
		is_training: bool Tensor variable

	Returns:
		Variable tensor
	"""
	with tf.variable_scope(scope) as sc:
		num_in_channels = inputs.get_shape()[-1].value
		kernel_shape = [kernel_size,
										num_in_channels, num_output_channels]
		kernel = _variable_with_weight_decay('weights',
																				 shape=kernel_shape,
																				 use_xavier=use_xavier,
																				 stddev=stddev,
																				 wd=weight_decay)
		outputs = tf.nn.conv1d(inputs, kernel,
													 stride=stride,
													 padding=padding)
		biases = _variable_on_cpu('biases', [num_output_channels],
															tf.constant_initializer(0.0))
		outputs = tf.nn.bias_add(outputs, biases)

		if bn:
			outputs = batch_norm_for_conv1d(outputs, is_training,
																			bn_decay=bn_decay, scope='bn', is_dist=is_dist)

		if activation_fn is not None:
			outputs = activation_fn(outputs)
		return outputs




def conv2d(inputs,
			num_output_channels,
			kernel_size,
			scope,
			stride=[1, 1],
			padding='SAME',
			use_xavier=True,
			stddev=1e-3,
			weight_decay=0.0,
			activation_fn=tf.nn.relu,
			bn=False,
			bn_decay=None,
			is_training=None,
			is_dist=False):
	""" 2D convolution with non-linear operation.

	Args:
		inputs: 4-D tensor variable BxHxWxC
		num_output_channels: int
		kernel_size: a list of 2 ints
		scope: string
		stride: a list of 2 ints
		padding: 'SAME' or 'VALID'
		use_xavier: bool, use xavier_initializer if true
		stddev: float, stddev for truncated_normal init
		weight_decay: float
		activation_fn: function
		bn: bool, whether to use batch norm
		bn_decay: float or float tensor variable in [0,1]
		is_training: bool Tensor variable

	Returns:
		Variable tensor
	"""
	with tf.variable_scope(scope) as sc:
			kernel_h, kernel_w = kernel_size
			num_in_channels = inputs.get_shape()[-1].value
			kernel_shape = [kernel_h, kernel_w,
											num_in_channels, num_output_channels]
			kernel = _variable_with_weight_decay('weights',
																					 shape=kernel_shape,
																					 use_xavier=use_xavier,
																					 stddev=stddev,
																					 wd=weight_decay)
			stride_h, stride_w = stride
			outputs = tf.nn.conv2d(inputs, kernel,
														 [1, stride_h, stride_w, 1],
														 padding=padding)
			biases = _variable_on_cpu('biases', [num_output_channels],
																tf.constant_initializer(0.0))
			outputs = tf.nn.bias_add(outputs, biases)

			if bn:
				outputs = batch_norm_for_conv2d(outputs, is_training,
																				bn_decay=bn_decay, scope='bn', is_dist=is_dist)

			if activation_fn is not None:
				outputs = activation_fn(outputs)
			return outputs

def conv2d_nobias(inputs,
           num_output_channels,
           kernel_size,
           scope,
           stride=[1, 1],
           padding='SAME',
           use_xavier=True,
           stddev=1e-3,
           weight_decay=0.0,
           activation_fn=tf.nn.relu,
           bn=False,
           bn_decay=None,
           is_training=None,
           is_dist=False):
    """ 2D convolution with non-linear operation.

      Args:
        inputs: 4-D tensor variable BxHxWxC
        num_output_channels: int
        kernel_size: a list of 2 ints
        scope: string
        stride: a list of 2 ints
        padding: 'SAME' or 'VALID'
        use_xavier: bool, use xavier_initializer if true
        stddev: float, stddev for truncated_normal init
        weight_decay: float
        activation_fn: function
        bn: bool, whether to use batch norm
        bn_decay: float or float tensor variable in [0,1]
        is_training: bool Tensor variable

      Returns:
        Variable tensor
      """
    with tf.variable_scope(scope) as sc:
        kernel_h, kernel_w = kernel_size
        num_in_channels = inputs.get_shape()[-1].value
        kernel_shape = [kernel_h, kernel_w,
                        num_in_channels, num_output_channels]
        kernel = _variable_with_weight_decay('weights',
                                             shape=kernel_shape,
                                             use_xavier=use_xavier,
                                             stddev=stddev,
                                             wd=weight_decay)
        stride_h, stride_w = stride
        outputs = tf.nn.conv2d(inputs, kernel,
                               [1, stride_h, stride_w, 1],
                               padding=padding)

        if bn:
            outputs = batch_norm_for_conv2d(outputs, is_training,
                                            bn_decay=bn_decay, scope='bn', is_dist=is_dist)

        if activation_fn is not None:
            outputs = activation_fn(outputs)
        return outputs


def conv2d_transpose(inputs,
					num_output_channels,
					kernel_size,
					scope,
					stride=[1, 1],
					padding='SAME',
					use_xavier=True,
					stddev=1e-3,
					weight_decay=0.0,
					activation_fn=tf.nn.relu,
					bn=False,
					bn_decay=None,
					is_training=None,
					is_dist=False):
	""" 2D convolution transpose with non-linear operation.

	Args:
		inputs: 4-D tensor variable BxHxWxC
		num_output_channels: int
		kernel_size: a list of 2 ints
		scope: string
		stride: a list of 2 ints
		padding: 'SAME' or 'VALID'
		use_xavier: bool, use xavier_initializer if true
		stddev: float, stddev for truncated_normal init
		weight_decay: float
		activation_fn: function
		bn: bool, whether to use batch norm
		bn_decay: float or float tensor variable in [0,1]
		is_training: bool Tensor variable

	Returns:
		Variable tensor

	Note: conv2d(conv2d_transpose(a, num_out, ksize, stride), a.shape[-1], ksize, stride) == a
	"""
	with tf.variable_scope(scope) as sc:
			kernel_h, kernel_w = kernel_size
			num_in_channels = inputs.get_shape()[-1].value
			kernel_shape = [kernel_h, kernel_w,
											num_output_channels, num_in_channels] # reversed to conv2d
			kernel = _variable_with_weight_decay('weights',
																					 shape=kernel_shape,
																					 use_xavier=use_xavier,
																					 stddev=stddev,
																					 wd=weight_decay)
			stride_h, stride_w = stride
			
			# from slim.convolution2d_transpose
			def get_deconv_dim(dim_size, stride_size, kernel_size, padding):
					dim_size *= stride_size

					if padding == 'VALID' and dim_size is not None:
						dim_size += max(kernel_size - stride_size, 0)
					return dim_size

			# caculate output shape
			batch_size = inputs.get_shape()[0].value
			height = inputs.get_shape()[1].value
			width = inputs.get_shape()[2].value
			out_height = get_deconv_dim(height, stride_h, kernel_h, padding)
			out_width = get_deconv_dim(width, stride_w, kernel_w, padding)
			output_shape = [batch_size, out_height, out_width, num_output_channels]

			outputs = tf.nn.conv2d_transpose(inputs, kernel, output_shape,
														 [1, stride_h, stride_w, 1],
														 padding=padding)
			biases = _variable_on_cpu('biases', [num_output_channels],
																tf.constant_initializer(0.0))
			outputs = tf.nn.bias_add(outputs, biases)

			if bn:
				outputs = batch_norm_for_conv2d(outputs, is_training,
																				bn_decay=bn_decay, scope='bn', is_dist=is_dist)

			if activation_fn is not None:
				outputs = activation_fn(outputs)
			return outputs

	 

def conv3d(inputs,
			num_output_channels,
			kernel_size,
			scope,
			stride=[1, 1, 1],
			padding='SAME',
			use_xavier=True,
			stddev=1e-3,
			weight_decay=0.0,
			activation_fn=tf.nn.relu,
			bn=False,
			bn_decay=None,
			is_training=None,
			is_dist=False):
	""" 3D convolution with non-linear operation.

	Args:
		inputs: 5-D tensor variable BxDxHxWxC
		num_output_channels: int
		kernel_size: a list of 3 ints
		scope: string
		stride: a list of 3 ints
		padding: 'SAME' or 'VALID'
		use_xavier: bool, use xavier_initializer if true
		stddev: float, stddev for truncated_normal init
		weight_decay: float
		activation_fn: function
		bn: bool, whether to use batch norm
		bn_decay: float or float tensor variable in [0,1]
		is_training: bool Tensor variable

	Returns:
		Variable tensor
	"""
	with tf.variable_scope(scope) as sc:
		kernel_d, kernel_h, kernel_w = kernel_size
		num_in_channels = inputs.get_shape()[-1].value
		kernel_shape = [kernel_d, kernel_h, kernel_w,
										num_in_channels, num_output_channels]
		kernel = _variable_with_weight_decay('weights',
											 shape=kernel_shape,
											 use_xavier=use_xavier,
											 stddev=stddev,
											 wd=weight_decay)
		stride_d, stride_h, stride_w = stride
		outputs = tf.nn.conv3d(inputs, kernel,
													 [1, stride_d, stride_h, stride_w, 1],
													 padding=padding)
		biases = _variable_on_cpu('biases', [num_output_channels],
															tf.constant_initializer(0.0))
		outputs = tf.nn.bias_add(outputs, biases)
		
		if bn:
			outputs = batch_norm_for_conv3d(outputs, is_training,
											bn_decay=bn_decay, scope='bn', is_dist=is_dist)

		if activation_fn is not None:
			outputs = activation_fn(outputs)
		return outputs

def fully_connected(inputs,
					num_outputs,
					scope,
					use_xavier=True,
					stddev=1e-3,
					weight_decay=0.0,
					activation_fn=tf.nn.relu,
					bn=False,
					bn_decay=None,
					is_training=None,
					is_dist=False):
	""" Fully connected layer with non-linear operation.
	
	Args:
		inputs: 2-D tensor BxN
		num_outputs: int
	
	Returns:
		Variable tensor of size B x num_outputs.
	"""
	with tf.variable_scope(scope) as sc:
		num_input_units = inputs.get_shape()[-1].value
		weights = _variable_with_weight_decay('weights',
											shape=[num_input_units, num_outputs],
											use_xavier=use_xavier,
											stddev=stddev,
											wd=weight_decay)
		outputs = tf.matmul(inputs, weights)
		biases = _variable_on_cpu('biases', [num_outputs],
														 tf.constant_initializer(0.0))
		outputs = tf.nn.bias_add(outputs, biases)
		 
		if bn:
			outputs = batch_norm_for_fc(outputs, is_training, bn_decay, 'bn', is_dist=is_dist)

		if activation_fn is not None:
			outputs = activation_fn(outputs)
		return outputs


def max_pool2d(inputs,
	kernel_size,
	scope,
	stride=[2, 2],
	padding='VALID'):
	""" 2D max pooling.

	Args:
		inputs: 4-D tensor BxHxWxC
		kernel_size: a list of 2 ints
		stride: a list of 2 ints
	
	Returns:
		Variable tensor
	"""
	with tf.variable_scope(scope) as sc:
		kernel_h, kernel_w = kernel_size
		stride_h, stride_w = stride
		outputs = tf.nn.max_pool(inputs,
		ksize=[1, kernel_h, kernel_w, 1],
		strides=[1, stride_h, stride_w, 1],
		padding=padding,
		name=sc.name)
		return outputs

def avg_pool2d(inputs,
				kernel_size,
				scope,
				stride=[2, 2],
				padding='VALID'):
	""" 2D avg pooling.

	Args:
		inputs: 4-D tensor BxHxWxC
		kernel_size: a list of 2 ints
		stride: a list of 2 ints
	
	Returns:
		Variable tensor
	"""
	with tf.variable_scope(scope) as sc:
		kernel_h, kernel_w = kernel_size
		stride_h, stride_w = stride
		outputs = tf.nn.avg_pool(inputs,
		ksize=[1, kernel_h, kernel_w, 1],
		strides=[1, stride_h, stride_w, 1],
		padding=padding,
		name=sc.name)
		return outputs


def max_pool3d(inputs,
			kernel_size,
			scope,
			stride=[2, 2, 2],
			padding='VALID'):
	""" 3D max pooling.

	Args:
		inputs: 5-D tensor BxDxHxWxC
		kernel_size: a list of 3 ints
		stride: a list of 3 ints
	
	Returns:
		Variable tensor
	"""
	with tf.variable_scope(scope) as sc:
		kernel_d, kernel_h, kernel_w = kernel_size
		stride_d, stride_h, stride_w = stride
		outputs = tf.nn.max_pool3d(inputs,
					ksize=[1, kernel_d, kernel_h, kernel_w, 1],
					strides=[1, stride_d, stride_h, stride_w, 1],
					padding=padding,
					name=sc.name)
		return outputs

def avg_pool3d(inputs,
							 kernel_size,
							 scope,
							 stride=[2, 2, 2],
							 padding='VALID'):
	""" 3D avg pooling.

	Args:
		inputs: 5-D tensor BxDxHxWxC
		kernel_size: a list of 3 ints
		stride: a list of 3 ints
	
	Returns:
		Variable tensor
	"""
	with tf.variable_scope(scope) as sc:
		kernel_d, kernel_h, kernel_w = kernel_size
		stride_d, stride_h, stride_w = stride
		outputs = tf.nn.avg_pool3d(inputs,
						ksize=[1, kernel_d, kernel_h, kernel_w, 1],
						strides=[1, stride_d, stride_h, stride_w, 1],
						padding=padding,
						name=sc.name)
		return outputs





def batch_norm_template(inputs, is_training, scope, moments_dims, bn_decay):
	""" Batch normalization on convolutional maps and beyond...
	Ref.: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow
	
	Args:
			inputs:        Tensor, k-D input ... x C could be BC or BHWC or BDHWC
			is_training:   boolean tf.Varialbe, true indicates training phase
			scope:         string, variable scope
			moments_dims:  a list of ints, indicating dimensions for moments calculation
			bn_decay:      float or float tensor variable, controling moving average weight
	Return:
			normed:        batch-normalized maps
	"""
	with tf.variable_scope(scope) as sc:
		num_channels = inputs.get_shape()[-1].value
		beta = tf.Variable(tf.constant(0.0, shape=[num_channels]),
											 name='beta', trainable=True)
		gamma = tf.Variable(tf.constant(1.0, shape=[num_channels]),
												name='gamma', trainable=True)
		batch_mean, batch_var = tf.nn.moments(inputs, moments_dims, name='moments')
		decay = bn_decay if bn_decay is not None else 0.9
		ema = tf.train.ExponentialMovingAverage(decay=decay)
		# Operator that maintains moving averages of variables.
		ema_apply_op = tf.cond(is_training,
													 lambda: ema.apply([batch_mean, batch_var]),
													 lambda: tf.no_op())
		
		# Update moving average and return current batch's avg and var.
		def mean_var_with_update():
			with tf.control_dependencies([ema_apply_op]):
				return tf.identity(batch_mean), tf.identity(batch_var)
		
		# ema.average returns the Variable holding the average of var.
		mean, var = tf.cond(is_training,
												mean_var_with_update,
												lambda: (ema.average(batch_mean), ema.average(batch_var)))
		normed = tf.nn.batch_normalization(inputs, mean, var, beta, gamma, 1e-3)
	return normed


def batch_norm_dist_template(inputs, is_training, scope, moments_dims, bn_decay):
	""" The batch normalization for distributed training.
	Args:
			inputs:        Tensor, k-D input ... x C could be BC or BHWC or BDHWC
			is_training:   boolean tf.Varialbe, true indicates training phase
			scope:         string, variable scope
			moments_dims:  a list of ints, indicating dimensions for moments calculation
			bn_decay:      float or float tensor variable, controling moving average weight
	Return:
			normed:        batch-normalized maps
	"""
	with tf.variable_scope(scope) as sc:
		num_channels = inputs.get_shape()[-1].value
		beta = _variable_on_cpu('beta', [num_channels], initializer=tf.zeros_initializer())
		gamma = _variable_on_cpu('gamma', [num_channels], initializer=tf.ones_initializer())

		pop_mean = _variable_on_cpu('pop_mean', [num_channels], initializer=tf.zeros_initializer(), trainable=False)
		pop_var = _variable_on_cpu('pop_var', [num_channels], initializer=tf.ones_initializer(), trainable=False)

		def train_bn_op():
			batch_mean, batch_var = tf.nn.moments(inputs, moments_dims, name='moments')
			decay = bn_decay if bn_decay is not None else 0.9
			train_mean = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1 - decay)) 
			train_var = tf.assign(pop_var, pop_var * decay + batch_var * (1 - decay))
			with tf.control_dependencies([train_mean, train_var]):
				return tf.nn.batch_normalization(inputs, batch_mean, batch_var, beta, gamma, 1e-3)

		def test_bn_op():
			return tf.nn.batch_normalization(inputs, pop_mean, pop_var, beta, gamma, 1e-3)

		normed = tf.cond(is_training,
										 train_bn_op,
										 test_bn_op)
		return normed



def batch_norm_for_fc(inputs, is_training, bn_decay, scope, is_dist=False):
	""" Batch normalization on FC data.
	
	Args:
			inputs:      Tensor, 2D BxC input
			is_training: boolean tf.Varialbe, true indicates training phase
			bn_decay:    float or float tensor variable, controling moving average weight
			scope:       string, variable scope
			is_dist:     true indicating distributed training scheme
	Return:
			normed:      batch-normalized maps
	"""
	if is_dist:
		return batch_norm_dist_template(inputs, is_training, scope, [0,], bn_decay)
	else:
		return batch_norm_template(inputs, is_training, scope, [0,], bn_decay)


def batch_norm_for_conv1d(inputs, is_training, bn_decay, scope, is_dist=False):
	""" Batch normalization on 1D convolutional maps.
	
	Args:
			inputs:      Tensor, 3D BLC input maps
			is_training: boolean tf.Varialbe, true indicates training phase
			bn_decay:    float or float tensor variable, controling moving average weight
			scope:       string, variable scope
			is_dist:     true indicating distributed training scheme
	Return:
			normed:      batch-normalized maps
	"""
	if is_dist:
		return batch_norm_dist_template(inputs, is_training, scope, [0,1], bn_decay)
	else:
		return batch_norm_template(inputs, is_training, scope, [0,1], bn_decay)



	
def batch_norm_for_conv2d(inputs, is_training, bn_decay, scope, is_dist=False):
	""" Batch normalization on 2D convolutional maps.
	
	Args:
			inputs:      Tensor, 4D BHWC input maps
			is_training: boolean tf.Varialbe, true indicates training phase
			bn_decay:    float or float tensor variable, controling moving average weight
			scope:       string, variable scope
			is_dist:     true indicating distributed training scheme
	Return:
			normed:      batch-normalized maps
	"""
	if is_dist:
		return batch_norm_dist_template(inputs, is_training, scope, [0,1,2], bn_decay)
	else:
		return batch_norm_template(inputs, is_training, scope, [0,1,2], bn_decay)



def batch_norm_for_conv3d(inputs, is_training, bn_decay, scope, is_dist=False):
	""" Batch normalization on 3D convolutional maps.
	
	Args:
			inputs:      Tensor, 5D BDHWC input maps
			is_training: boolean tf.Varialbe, true indicates training phase
			bn_decay:    float or float tensor variable, controling moving average weight
			scope:       string, variable scope
			is_dist:     true indicating distributed training scheme
	Return:
			normed:      batch-normalized maps
	"""
	if is_dist:
		return batch_norm_dist_template(inputs, is_training, scope, [0,1,2,3], bn_decay)
	else:
		return batch_norm_template(inputs, is_training, scope, [0,1,2,3], bn_decay)


def dropout(inputs,
			is_training,
			scope,
			keep_prob=0.5,
			noise_shape=None):
	""" Dropout layer.

	Args:
		inputs: tensor
		is_training: boolean tf.Variable
		scope: string
		keep_prob: float in [0,1]
		noise_shape: list of ints

	Returns:
		tensor variable
	"""
	with tf.variable_scope(scope) as sc:
		outputs = tf.cond(is_training,
		lambda: tf.nn.dropout(inputs, keep_prob, noise_shape),
		lambda: inputs)
		return outputs

def gather_labels(input_labels, index):
	batch_size = input_labels.get_shape()[0].value
	num_points = input_labels.get_shape()[1].value

	idx_ = tf.range(batch_size) * num_points
	idx_ = tf.reshape(idx_, [batch_size, 1])

	input_labels_flat = tf.reshape(input_labels, [-1, 1])
	selected_labels = tf.gather(input_labels_flat, index + idx_)

	selected_labels = tf.squeeze(selected_labels)
	if batch_size == 1:
		selected_labels = tf.expand_dims(selected_labels, 0)

	return selected_labels


def pairwise_distance(point_cloud):
	"""Compute pairwise distance of a point cloud.

	Args:
		point_cloud: tensor (batch_size, num_points, num_dims)

	Returns:
		pairwise distance: (batch_size, num_points, num_points)
	"""
	og_batch_size = point_cloud.get_shape().as_list()[0]
	num_points = point_cloud.get_shape().as_list()[1]

	point_cloud = tf.squeeze(point_cloud)
	if og_batch_size == 1:
		point_cloud = tf.expand_dims(point_cloud, 0)

	if num_points == 1:
		point_cloud = tf.expand_dims(point_cloud, 1)
		
	point_cloud_transpose = tf.transpose(point_cloud, perm=[0, 2, 1])
	point_cloud_inner = tf.matmul(point_cloud, point_cloud_transpose)
	point_cloud_inner = -2*point_cloud_inner
	point_cloud_square = tf.reduce_sum(tf.square(point_cloud), axis=-1, keepdims=True)
	point_cloud_square_tranpose = tf.transpose(point_cloud_square, perm=[0, 2, 1])
	return point_cloud_square + point_cloud_inner + point_cloud_square_tranpose


def knn(adj_matrix, k=20):
	"""Get KNN based on the pairwise distance.
	Args:
		pairwise distance: (batch_size, num_points, num_points)
		k: int

	Returns:
		nearest neighbors: (batch_size, num_points, k)
	"""
	neg_adj = -adj_matrix
	_, nn_idx = tf.nn.top_k(neg_adj, k=k)
	return nn_idx


def get_edge_feature(point_cloud, nn_idx, k=20):
	"""Construct edge feature for each point
	Args:
		point_cloud: (batch_size, num_points, 1, num_dims)
		nn_idx: (batch_size, num_points, k)
		k: int

	Returns:
		edge features: (batch_size, num_points, k, num_dims)
	"""
	og_batch_size = point_cloud.get_shape().as_list()[0]
	point_cloud = tf.squeeze(point_cloud)
	if og_batch_size == 1:
		point_cloud = tf.expand_dims(point_cloud, 0)

	point_cloud_central = point_cloud

	point_cloud_shape = point_cloud.get_shape()
	batch_size = point_cloud_shape[0].value
	num_points = point_cloud_shape[1].value
	num_dims = point_cloud_shape[2].value

	idx_ = tf.range(batch_size) * num_points
	idx_ = tf.reshape(idx_, [batch_size, 1, 1]) 

	point_cloud_flat = tf.reshape(point_cloud, [-1, num_dims])
	point_cloud_neighbors = tf.gather(point_cloud_flat, nn_idx+idx_)
	point_cloud_central = tf.expand_dims(point_cloud_central, axis=-2)

	point_cloud_central = tf.tile(point_cloud_central, [1, 1, k, 1])

	edge_feature = tf.concat([point_cloud_central, point_cloud_neighbors-point_cloud_central], axis=-1)
	return edge_feature



def attention_conv(xyz_input, feature_input, output_dim, nn_idx, k, scope, activation=tf.nn.relu, bn=False, bn_decay=None,
	is_training=None, is_dist=False):


	with tf.variable_scope(scope) as sc:


		feature_input_neighbors = group_point(feature_input, nn_idx)

		feature_input_central = tf.expand_dims(feature_input, axis=-2)

		central = conv2d_nobias(feature_input_central, output_dim, [1, 1], padding='VALID', stride=[1, 1], bn=True,
											is_training=is_training, scope="1",
											bn_decay=bn_decay, is_dist=is_dist)

		input_feature_tiled = tf.tile(feature_input_central, [1, 1, k, 1])
		edge_feature = feature_input_neighbors - input_feature_tiled
		print(567, edge_feature.shape)

		feature = tf.concat([input_feature_tiled, edge_feature],axis = -1)

		edge_feature = conv2d(edge_feature, output_dim, [1, 1], padding='VALID', stride=[1, 1],
									  bn=True, is_training=is_training, scope="2",
									  bn_decay=bn_decay, is_dist=is_dist)

		self_attention = conv2d(central, 1, [1, 1], padding='VALID', stride=[1, 1], bn=True,
										is_training=is_training, scope="3",
										bn_decay=bn_decay, is_dist=is_dist)
		neibor_attention = conv2d(edge_feature, 1, [1, 1], padding='VALID', stride=[1, 1], bn=True,
										  is_training=is_training, scope="4",
										  bn_decay=bn_decay, is_dist=is_dist)

		feature = conv2d(feature, output_dim, [1, 1], padding='VALID', stride=[1, 1],
								 bn=True, is_training=is_training, scope="5",
								 bn_decay=bn_decay, is_dist=is_dist)


		logits = self_attention + neibor_attention

		logits = tf.transpose(logits, [0, 1, 3, 2])

		coefs = tf.nn.softmax(tf.nn.leaky_relu(logits))

		vals = tf.matmul(coefs, feature)

		ret = tf.contrib.layers.bias_add(vals)
		net = activation(ret)
		net = tf.squeeze(net, [2])

	return net


def attention_pooling(feature_input, point_input, num_samples, knn, scope, bn_decay, is_training, bn=True):


	with tf.variable_scope(scope) as sc:
		p_idx = farthest_point_sample(num_samples, point_input)

		new_xyz = gather_point(point_input, p_idx)
		new_feature = gather_point(feature_input, p_idx)


		feature_shape = new_xyz.get_shape()
		batch_size = feature_shape[0]
		num_points = int(feature_shape[1])
		num_out_channel = feature_input.get_shape()[-1].value


		nsample = int(min(knn, num_points))
		_, pn_idx = knn_point(nsample, point_input, new_xyz)

		grouped_points = group_point(feature_input, pn_idx)


		center_feature = group_point(feature_input, tf.expand_dims(p_idx, 2))
		edge_feature = grouped_points - tf.tile(center_feature, [1, 1, nsample, 1])
		new_points = edge_feature

		num_out_channel = new_points.get_shape()[-1].value
		feature_1 = conv2d(new_points, num_out_channel/2, [1, 1],
								  padding='VALID', stride=[1, 1],
								  bn=True, is_training=is_training,
								  scope='conv_output_a', bn_decay=bn_decay)

		feature_2 = conv2d(new_points, num_out_channel/2, [1, 1],
								  padding='VALID', stride=[1, 1],
								  bn=True, is_training=is_training,
								  scope='conv_output_b', bn_decay=bn_decay)

		feature_2 = tf.transpose(feature_2, [0, 1, 3, 2])

		energy = tf.matmul(feature_1, feature_2)
		attention = tf.nn.softmax(energy, axis=-1)

		feature_3 = conv2d(new_points, num_out_channel, [1, 1],
								  padding='VALID', stride=[1, 1],
								  bn=True, is_training=is_training,
								  scope='conv_output_d', bn_decay=bn_decay)


		gamma = _variable_with_weight_decay('weight_patial',
													shape=[1],
													use_xavier=True,
													stddev=1e-3,
													wd=0.0)

		feature_SA = tf.matmul(attention, feature_3)
		feature_SA = feature_SA * gamma + new_points

		feature_4 = tf.transpose(new_points, [0, 1, 3, 2])
		energy = tf.matmul(feature_4, new_points)

		D = tf.reduce_max(energy, -1)
		D = tf.expand_dims(D, -1)

		energy_new = tf.tile(D, multiples=[1, 1, 1, energy.shape[3]]) - energy
		attention = tf.nn.softmax(energy_new, axis=-1)

		feature_CA = tf.matmul(new_points, attention)

		gamma2 = _variable_with_weight_decay('weightsgamma2m',
													 shape=[1],
													 use_xavier=True,
													 stddev=1e-3,
													 wd=0.0)
		feature_CA = feature_CA * gamma2 + new_points

		output = feature_SA + feature_CA

		output = tf.reduce_max(output, axis=[2], keep_dims=True, name='maxpool2')
		output = tf.concat([center_feature, output], axis=-1)

		net = tf.squeeze(output, [2])  # (batch_size, npoints, mlp2[-1])

	return net, p_idx, pn_idx, new_xyz



def three_nn_upsampling(target_points, source_points):

	dist, idx = three_nn(target_points, source_points)
	dist = tf.maximum(dist, 1e-10)
	norm = tf.reduce_sum((1.0/dist), axis=2, keepdims=True)
	norm = tf.tile(norm, [1, 1, 3])
	weight = (1.0 / dist) / norm

	return idx, weight