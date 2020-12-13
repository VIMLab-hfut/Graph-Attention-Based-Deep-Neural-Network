import tensorflow as tf
import numpy as np
import math
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
from pointSIFT_op import pointSIFT_select, pointSIFT_select_four, pointSIFT_select_two
from tf_grouping import query_ball_point, group_point, knn_point

import tf_util
import extra_loss


def placeholder_inputs(batch_size, num_point, normal_flag=False):
	if normal_flag:
		pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 6))
	else:
		pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
	labels_pl = tf.placeholder(tf.int32, shape=(batch_size))
	return pointclouds_pl, labels_pl

def get_model(point_input, is_training, pfs_flag=False, bn_decay=None):
	""" Classification PointNet, input is BxNxC, output Bx40 """
	batch_size = point_input.get_shape()[0].value
	num_point1 = point_input.get_shape()[1].value
	num_point2 = int(np.floor(num_point1 / 4.0))
	num_point3 = int(np.floor(num_point2 / 4.0))

	num_features = point_input.get_shape()[2].value

	end_points = {}

	point_cloud1 = point_input


	k = 32
	nn_idx = pointSIFT_select_four(point_cloud1, 0.2)
	net1_1 = tf_util.attention_conv(point_cloud1, point_input, 64, nn_idx, k, scope='conv_1_1', bn=True,
									 bn_decay=bn_decay, is_training=is_training)
	net1_2 = tf_util.attention_conv(point_cloud1, net1_1, 64, nn_idx, k, scope='conv_1_2', bn=True,
									 bn_decay=bn_decay,
									 is_training=is_training)

	k = 30
	net, p1_idx, pn_idx, point_cloud2 = tf_util.attention_pooling(net1_2, point_cloud1, num_point2, k, scope='12', bn_decay=bn_decay, is_training=is_training)

	net1_1 = tf.squeeze(tf.reduce_max(group_point(net1_1, pn_idx), axis=-2, keepdims=True))
	net1_2 = net


	k = 16
	nn_idx = pointSIFT_select_two(point_cloud2, 0.4)
	net2_1 = tf_util.attention_conv(point_cloud2, net, 128, nn_idx, k, scope='conv_2_1', bn=True,
									 bn_decay=bn_decay,
									 is_training=is_training)
	net2_2 = tf_util.attention_conv(point_cloud2, net2_1, 128, nn_idx, k, scope='conv_2_2', bn=True,
									 bn_decay=bn_decay,
									 is_training=is_training)



	k = 30
	net, p2_idx, pn_idx, point_cloud3 = tf_util.attention_pooling(net2_2, point_cloud2, num_point3, k, scope='13', bn_decay=bn_decay, is_training=is_training)
	print(6666, net1_1.shape)
	net1_1 = tf.reduce_max(group_point(net1_1, pn_idx), axis=-2, keepdims=True)
	net1_2 = tf.reduce_max(group_point(net1_2, pn_idx), axis=-2, keepdims=True)
	net2_1 = tf.reduce_max(group_point(net2_1, pn_idx), axis=-2, keepdims=True)
	net2_2 = net


	k = 16
	nn_idx = pointSIFT_select_two(point_cloud3, 0.6)
	net3_1 = tf_util.attention_conv(point_cloud3, net, 256, nn_idx, k, scope='conv_3_1', bn=True,
									 bn_decay=bn_decay,
									 is_training=is_training)
	net3_2 = tf_util.attention_conv(point_cloud3, net3_1, 256, nn_idx, k, scope='conv_3_2', bn=True,
									 bn_decay=bn_decay,
									 is_training=is_training)

	net3_1 = tf.expand_dims(net3_1, axis=-2)
	net3_2 = tf.expand_dims(net3_2, axis=-2)
	net2_2 = tf.expand_dims(net2_2, axis=-2)


	net = tf.concat([net1_1, net1_2, net2_1, net2_2, net3_1, net3_2], axis=-1)
	net = tf_util.conv2d(net, 1024, [1, 1],
						 padding='VALID', stride=[1, 1], activation_fn=tf.nn.relu,
						 bn=True, is_training=is_training,
						 scope='agg', bn_decay=bn_decay)

	net = tf.reduce_max(net, axis=1, keepdims=True)

	net = tf.reshape(net, [batch_size, -1])

	end_points['embedding'] = net

	net = tf_util.fully_connected(net, 512, bn=True, is_training=is_training,
								  scope='fc1', bn_decay=bn_decay)
	net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training, scope='dropout1')
	net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training,
								  scope='fc2', bn_decay=bn_decay)
	net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training, scope='dropout2')
	net = tf_util.fully_connected(net, 40, activation_fn=None, scope='fc3')

	return net, end_points






def get_loss(pred, label, end_points, mmd_flag):
	""" pred: B*NUM_CLASSES,
	  label: B, """
	labels = tf.one_hot(indices=label, depth=40)
	loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=pred, label_smoothing=0.2)
	classify_loss = tf.reduce_mean(loss)
	tf.summary.scalar('cls loss', classify_loss)

	loss = classify_loss

	if mmd_flag > 0:
		batch_size = end_points['embedding'].get_shape()[0].value
		feature_size = end_points['embedding'].get_shape()[1].value

		true_samples = tf.random_normal(tf.stack([batch_size, feature_size]))
		mmd_loss = extra_loss.compute_mmd(end_points['embedding'], true_samples)
		mmd_loss = mmd_loss * mmd_flag
		tf.summary.scalar('mmd loss', mmd_loss)

		loss = loss + mmd_loss

	tf.add_to_collection('losses', loss)
	return loss














