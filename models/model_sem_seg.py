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
from tf_sampling import farthest_point_sample, gather_point
from tf_grouping import query_ball_point, group_point, knn_point
from tf_interpolate import three_nn, three_interpolate

import tf_util
import extra_loss


def placeholder_inputs(batch_size, num_point):
	pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 9))
	labels_pl = tf.placeholder(tf.int32, shape=(batch_size, num_point))
	return pointclouds_pl, labels_pl

def get_model(point_input, labels_pl, is_training, bn_decay=None, coarse_flag=0,
				mmd_flag=0, pfs_flag=False, fully_concate=False):
	batch_size = point_input.get_shape()[0].value
	num_point1 = point_input.get_shape()[1].value
	num_point2 = int(np.floor(num_point1 / 4.0))
	num_point3 = int(np.floor(num_point2 / 4.0))

	end_points = {}

	activation_fn = tf.nn.relu


	point_cloud1 = tf.slice(point_input, [0, 0, 0], [-1, -1, 3])



	k = 16
	nn_idx = pointSIFT_select_two(point_cloud1, 0.2)
	net1_1 = tf_util.attention_conv(point_cloud1, point_input, 64, nn_idx, k, scope='conv_1_1', bn=False,
									 bn_decay=bn_decay, is_training=is_training)
	net1_2 = tf_util.attention_conv(point_cloud1, net1_1, 64, nn_idx, k, scope='conv_1_2', bn=False, bn_decay=bn_decay,
									 is_training=is_training)

	k = 20
	net, p1_idx, pn_idx, point_cloud2 = tf_util.attention_pooling(net1_2, point_cloud1, num_point2, k, scope='12',
																   bn_decay=bn_decay, is_training=is_training)

	k = 8
	nn_idx = pointSIFT_select(point_cloud2, 0.4)
	net2_1 = tf_util.attention_conv(point_cloud2, net, 64, nn_idx, k, scope='conv_2_1', bn=False, bn_decay=bn_decay,
									 is_training=is_training)
	net2_2 = tf_util.attention_conv(point_cloud2, net2_1, 64, nn_idx, k, scope='conv_2_2', bn=False, bn_decay=bn_decay,
									 is_training=is_training)

	k = 20
	net, p2_idx, pn_idx, point_cloud3 = tf_util.attention_pooling(net2_2, point_cloud2, num_point3, k, scope='13',
																   bn_decay=bn_decay, is_training=is_training)

	k = 8
	nn_idx = pointSIFT_select(point_cloud3, 0.7)
	net3_1 = tf_util.attention_conv(point_cloud3, net, 128, nn_idx, k, scope='conv_3_1', bn=False,
									 bn_decay=bn_decay,
									 is_training=is_training)
	net3_2 = tf_util.attention_conv(point_cloud3, net3_1, 128, nn_idx, k, scope='conv_3_2', bn=False,
									 bn_decay=bn_decay,
									 is_training=is_training)
	net3_2 = tf.expand_dims(net3_2, 2)


	net = tf_util.conv2d(net3_2, 1024, [1, 1],
						 padding='VALID', stride=[1, 1], activation_fn=activation_fn,
						 bn=False, is_training=is_training,
						 scope='encoder', bn_decay=bn_decay)

	net = tf.reduce_max(net, axis=1, keepdims=True)
	net = tf.reshape(net, [batch_size, -1])
	net = tf_util.fully_connected(net, 512, bn=False, is_training=is_training, activation_fn=None,
								  scope='rg1', bn_decay=bn_decay)

	if mmd_flag > 0:
		end_points['embedding'] = net
	else:
		end_points['embedding'] = None

	net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training, scope='rgdp1')

	global_feature_size = 1024
	net = tf_util.fully_connected(net, global_feature_size, bn=False, is_training=is_training,
								  activation_fn=activation_fn,
								  scope='rg2', bn_decay=bn_decay)
	net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training, scope='rgdp2')
	net = tf.reshape(net, [batch_size, 1, 1, global_feature_size])

	net = tf.tile(net, [1, num_point3, 1, 1])

	net = tf.concat([net, net3_2], axis=-1)

	net = tf_util.conv2d(net, 512, [1, 1],
						 padding='VALID', stride=[1, 1], activation_fn=activation_fn,
						 bn=False, is_training=is_training,
						 scope='decoder', bn_decay=bn_decay)

	if coarse_flag > 0:
		coarse_net = tf.squeeze(net, [2])
		coarse_net = tf_util.conv1d(coarse_net, 128, 1, padding='VALID', bn=False, activation_fn=activation_fn,
									is_training=is_training, scope='coarse_fc1', bn_decay=bn_decay)
		coarse_net = tf_util.dropout(coarse_net, keep_prob=0.5, is_training=is_training, scope='cdp1')
		coarse_net = tf_util.conv1d(coarse_net, 50, 1, padding='VALID', activation_fn=None, scope='coarse_fc2')

		coarse_labels_pl = tf_util.gather_labels(labels_pl, p1_idx)
		coarse_labels_pl = tf_util.gather_labels(coarse_labels_pl, p2_idx)

		end_points['coarse_pred'] = coarse_net
		end_points['coarse_label'] = coarse_labels_pl

	else:
		end_points['coarse_pred'] = None
		end_points['coarse_label'] = None

	k = 8
	nn_idx = pointSIFT_select(point_cloud3, 0.7)
	net3_2 = tf_util.attention_conv(point_cloud3, tf.squeeze(net,[2]), 128, nn_idx, k, scope='conv_4_1', bn=False,
									 bn_decay=bn_decay,
									 is_training=is_training)

	net = tf.concat([net3_2, net3_1], axis=-1)


	net3_1 = tf_util.attention_conv(point_cloud3, net, 128, nn_idx, k, scope='conv_4_2', bn=False,
									 bn_decay=bn_decay,
									 is_training=is_training)



	idx, weight = tf_util.three_nn_upsampling(point_cloud2, point_cloud3)
	net3_1 = three_interpolate(net3_1, idx, weight)
	if fully_concate:
		net3_2 = three_interpolate(net3_2, idx, weight)

	net = tf.concat([net3_1, net2_2], axis=-1)
	k = 8
	nn_idx = pointSIFT_select(point_cloud2, 0.4)
	net2_2 = tf_util.attention_conv(point_cloud2, net, 64, nn_idx, k, scope='conv_5_1', bn=False,
									 bn_decay=bn_decay,
									 is_training=is_training)

	net = tf.concat([net2_2, net2_1], axis=-1)


	net2_1 = tf_util.attention_conv(point_cloud2, net, 64, nn_idx, k, scope='conv_5_2', bn=False,
									 bn_decay=bn_decay,
									 is_training=is_training)

	idx, weight = tf_util.three_nn_upsampling(point_cloud1, point_cloud2)
	net2_1 = three_interpolate(net2_1, idx, weight)
	net3_1 = three_interpolate(net3_1, idx, weight)
	if fully_concate:
		net2_2 = three_interpolate(net2_2, idx, weight)
		net3_2 = three_interpolate(net3_2, idx, weight)

	net = tf.concat([net2_1, net1_2], axis=-1)
	k = 16
	nn_idx = pointSIFT_select_two(point_cloud1, 0.2)
	net1_2 = tf_util.attention_conv(point_cloud1, net, 64, nn_idx, k, scope='conv_6_1', bn=False,
									 bn_decay=bn_decay,
									 is_training=is_training)

	net = tf.concat([net1_2, net1_1], axis=-1)

	net1_1 = tf_util.attention_conv(point_cloud1, net, 64, nn_idx, k, scope='conv_6_2', bn=False,
									 bn_decay=bn_decay,
									 is_training=is_training)



	if fully_concate:
		net = tf.concat(
			[net1_1, net1_2, net2_1, net2_2, net3_1,
			 net3_2], axis=-1)
	else:
		net = tf.concat([net1_1, tf.expand_dims(net2_1, 2), tf.expand_dims(net3_1, 2)], axis=-1)

	net = tf_util.conv1d(net, 128, 1, padding='VALID', bn=False, activation_fn=activation_fn,
						 is_training=is_training, scope='fc1', bn_decay=bn_decay)
	net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training, scope='dp1')
	net = tf_util.conv1d(net, 13, 1, padding='VALID', activation_fn=None, scope='fc2')

	return net, end_points



def get_loss(pred, label, end_points, coarse_flag, mmd_flag):
	""" pred: BxNxC,
		label: BxN, """
	seg_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=label)
	seg_loss = tf.reduce_mean(seg_loss)
	tf.summary.scalar('seg loss', seg_loss)

	loss = seg_loss

	if coarse_flag > 0:
		coarse_seg_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=end_points['coarse_pred'], labels=end_points['coarse_label'])
		coarse_seg_loss = tf.reduce_mean(coarse_seg_loss)
		coarse_seg_loss = coarse_seg_loss * coarse_flag
		tf.summary.scalar('coarse seg loss', coarse_seg_loss)

		loss = loss + coarse_seg_loss

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

