import tensorflow as tf 
import numpy as np

import plot

import time
import math
import os
import sys

import plot

import DeepIoT_dropOut
import DeepIoT_rnnDrop

from har_tfrecord_util import input_pipeline_har

layers = tf.contrib.layers 

SEPCTURAL_SAMPLES = 10
FEATURE_DIM = SEPCTURAL_SAMPLES*6*2
CONV_LEN = 3
CONV_LEN_INTE = 3
CONV_LEN_LAST = 3
CONV_NUM = 64
CONV_MERGE_LEN = 8
CONV_MERGE_LEN2 = 6
CONV_MERGE_LEN3 = 4
CONV_NUM2 = 64
INTER_DIM = 120
OUT_DIM = 6
WIDE = 20
CONV_KEEP_PROB = 0.8
RNN_KEEP_PROB = 0.5

BATCH_SIZE = 64
TOTAL_ITER_NUM = 100000

REG_TERM = 1e-4


select = 'a'

metaDict = {'a':[119080, 1193], 'b':[116870, 1413], 'c':[116020, 1477]}
TRAIN_SIZE = metaDict[select][0]
EVAL_DATA_SIZE = metaDict[select][1]
EVAL_ITER_NUM = int(math.ceil(EVAL_DATA_SIZE / BATCH_SIZE))

prob_list_acc1 = tf.get_variable("prob_list_acc1", [1, 1, 1, CONV_NUM], tf.float32, tf.constant_initializer(CONV_KEEP_PROB), trainable=False)
prob_list_acc2 = tf.get_variable("prob_list_acc2", [1, 1, 1, CONV_NUM], tf.float32, tf.constant_initializer(CONV_KEEP_PROB), trainable=False)
prob_list_gyro1 = tf.get_variable("prob_list_gyro1", [1, 1, 1, CONV_NUM], tf.float32, tf.constant_initializer(CONV_KEEP_PROB), trainable=False)
prob_list_gyro2 = tf.get_variable("prob_list_gyro2", [1, 1, 1, CONV_NUM], tf.float32, tf.constant_initializer(CONV_KEEP_PROB), trainable=False)
prob_list_senIn = tf.get_variable("prob_list_senIn", [1, 1, 1, 1, CONV_NUM], tf.float32, tf.constant_initializer(CONV_KEEP_PROB), trainable=False)
prob_list_sen1 = tf.get_variable("prob_list_sen1", [1, 1, 1, 1, CONV_NUM2], tf.float32, tf.constant_initializer(CONV_KEEP_PROB), trainable=False)
prob_list_sen2 = tf.get_variable("prob_list_sen2", [1, 1, 1, 1, CONV_NUM2], tf.float32, tf.constant_initializer(CONV_KEEP_PROB), trainable=False)
prob_list_sen3 = tf.get_variable("prob_list_sen3", [1, 1, 1, 1, CONV_NUM2], tf.float32, tf.constant_initializer(CONV_KEEP_PROB), trainable=False)
prob_list_rnn1 = tf.get_variable("prob_list_rnn1", [1, INTER_DIM], tf.float32, tf.constant_initializer(RNN_KEEP_PROB), trainable=False)
prob_list_rnn2 = tf.get_variable("prob_list_rnn2", [1, INTER_DIM], tf.float32, tf.constant_initializer(RNN_KEEP_PROB), trainable=False)

sol_train = tf.Variable(0, dtype=tf.float32, trainable=False)
prun_thres = tf.get_variable("prun_thres", [], tf.float32, tf.constant_initializer(0.0), trainable=False)

prob_list_dict = {u'acc_conv1':prob_list_acc1, u'acc_conv2':prob_list_acc2, u'gyro_conv1':prob_list_gyro1, u'gyro_conv2':prob_list_gyro2,
					u'acc_conv3':prob_list_senIn, u'gyro_conv3':prob_list_senIn, u'sensor_conv1':prob_list_sen1, u'sensor_conv2':prob_list_sen2,
					u'sensor_conv3':prob_list_sen3, u'cell_0':prob_list_rnn1, u'cell_1':prob_list_rnn2}

org_dim_dict = {u'acc_conv1':CONV_NUM, u'acc_conv2':CONV_NUM, u'gyro_conv1':CONV_NUM, u'gyro_conv2':CONV_NUM,
					u'acc_conv3':CONV_NUM, u'gyro_conv3':CONV_NUM, u'sensor_conv1':CONV_NUM2, u'sensor_conv2':CONV_NUM2,
					u'sensor_conv3':CONV_NUM2, u'cell_0':INTER_DIM, u'cell_1':INTER_DIM}

###### Util Start
def dropOut_prun(drop_prob, prun_thres, sol_train):
	base_prob = 0.5
	pruned_drop_prob = tf.cond(sol_train > 0.5, lambda: tf.where(tf.less(drop_prob, prun_thres), tf.zeros_like(drop_prob), drop_prob), 
		lambda: tf.where(tf.less(drop_prob, prun_thres), drop_prob * base_prob, drop_prob))
	return pruned_drop_prob

def count_prun(prob_list_dict, prun_thres):
	left_num_dict = {}
	for layer_name in prob_list_dict.keys():
		prob_list = prob_list_dict[layer_name]
		pruned_idt = tf.where(tf.less(prob_list, prun_thres), tf.zeros_like(prob_list), tf.ones_like(prob_list))
		left_num = tf.reduce_sum(pruned_idt)
		left_num_dict[layer_name] = left_num
	return left_num_dict

def gen_cur_prun(sess, left_num_dict):
	cur_left_num = {}
	for layer_name in left_num_dict.keys():
		cur_left_num[layer_name] = sess.run(left_num_dict[layer_name])
	return cur_left_num

def compress_ratio(cur_left_num, org_dim_dict):
	layer_size_dict = {u'acc_conv1':2*3*CONV_LEN, u'acc_conv2':CONV_LEN_INTE, u'gyro_conv1':2*3*CONV_LEN, u'gyro_conv2':CONV_LEN_INTE,
						u'acc_conv3':CONV_LEN_LAST, u'gyro_conv3':CONV_LEN_LAST, u'sensor_conv1':2*CONV_MERGE_LEN, u'sensor_conv2':2*CONV_MERGE_LEN,
						u'sensor_conv3':2*CONV_MERGE_LEN, u'cell_0':8, u'cell_1':1}
	ord_list1 = [u'acc_conv1', u'acc_conv2', u'acc_conv3']
	ord_list2 = [u'gyro_conv1', u'gyro_conv2', u'gyro_conv3']
	ord_list3 = [u'sensor_conv1', u'sensor_conv2', u'sensor_conv3', u'cell_0', u'cell_1']

	org_size = 0
	comps_size = 0
	for idx, layer_name in enumerate(ord_list1):
		if idx == 0:
			org_size += layer_size_dict[layer_name]*1*org_dim_dict[layer_name] + org_dim_dict[layer_name]
			comps_size += layer_size_dict[layer_name]*1*cur_left_num[layer_name] + cur_left_num[layer_name]
		else:
			last_layer_name = ord_list1[idx-1]
			org_size += layer_size_dict[layer_name]*org_dim_dict[last_layer_name]*org_dim_dict[layer_name] + org_dim_dict[layer_name]
			comps_size += layer_size_dict[layer_name]*cur_left_num[last_layer_name]*cur_left_num[layer_name] + cur_left_num[layer_name]
	for idx, layer_name in enumerate(ord_list2):
		if idx == 0:
			org_size += layer_size_dict[layer_name]*1*org_dim_dict[layer_name] + org_dim_dict[layer_name]
			comps_size += layer_size_dict[layer_name]*1*cur_left_num[layer_name] + cur_left_num[layer_name]
		else:
			last_layer_name = ord_list1[idx-1]
			org_size += layer_size_dict[layer_name]*org_dim_dict[last_layer_name]*org_dim_dict[layer_name] + org_dim_dict[layer_name]
			comps_size += layer_size_dict[layer_name]*cur_left_num[last_layer_name]*cur_left_num[layer_name] + cur_left_num[layer_name]
	for idx, layer_name in enumerate(ord_list3):
		if idx == 0:
			last_layer_name = u'acc_conv3'
		else:
			last_layer_name = ord_list3[idx-1]
		if not 'cell' in layer_name:
			org_size += layer_size_dict[layer_name]*org_dim_dict[last_layer_name]*org_dim_dict[layer_name] + org_dim_dict[layer_name]
			comps_size += layer_size_dict[layer_name]*cur_left_num[last_layer_name]*cur_left_num[layer_name] + cur_left_num[layer_name]
		else:
			org_size += (layer_size_dict[layer_name]*org_dim_dict[last_layer_name] + org_dim_dict[layer_name])*3*org_dim_dict[layer_name]  + 3*org_dim_dict[layer_name]
			comps_size += (layer_size_dict[layer_name]*cur_left_num[last_layer_name] + cur_left_num[layer_name])*3*cur_left_num[layer_name]  + 3*cur_left_num[layer_name] 
	org_size += org_dim_dict[u'cell_1']*6 + 6
	comps_size += cur_left_num[u'cell_1']*6 + 6
	comps_ratio = float(comps_size)*100./org_size
	print 'Original Size:', org_size, 'Compressed Size:', comps_size, 'Left Ratio:', comps_ratio
	return comps_ratio

	
###### Util End


###### DeepSense Part Start
def batch_norm_layer(inputs, phase_train, scope=None):
	if phase_train:
		return layers.batch_norm(inputs, is_training=True, scale=True, 
			updates_collections=None, scope=scope)
	else:
		return layers.batch_norm(inputs, is_training=False, scale=True,
			updates_collections=None, scope=scope, reuse = True)

def deepSense(inputs, train, reuse=False, name='deepSense'):
	with tf.variable_scope(name, reuse=reuse) as scope:
		used = tf.sign(tf.reduce_max(tf.abs(inputs), reduction_indices=2)) #(BATCH_SIZE, WIDE)
		length = tf.reduce_sum(used, reduction_indices=1) #(BATCH_SIZE)
		length = tf.cast(length, tf.int64)

		mask = tf.sign(tf.reduce_max(tf.abs(inputs), reduction_indices=2, keep_dims=True))
		mask = tf.tile(mask, [1,1,INTER_DIM]) # (BATCH_SIZE, WIDE, INTER_DIM)
		avgNum = tf.reduce_sum(mask, reduction_indices=1) #(BATCH_SIZE, INTER_DIM)

		out_binary_mask = {}

		# inputs shape (BATCH_SIZE, WIDE, FEATURE_DIM)
		sensor_inputs = tf.expand_dims(inputs, axis=3)
		# sensor_inputs shape (BATCH_SIZE, WIDE, FEATURE_DIM, CHANNEL=1)
		acc_inputs, gyro_inputs = tf.split(sensor_inputs, num_or_size_splits=2, axis=2)

		acc_conv1 = layers.convolution2d(acc_inputs, CONV_NUM, kernel_size=[1, 2*3*CONV_LEN],
						stride=[1, 2*3], padding='VALID', activation_fn=None, data_format='NHWC', scope='acc_conv1')
		acc_conv1 = batch_norm_layer(acc_conv1, train, scope='acc_BN1')
		acc_conv1 = tf.nn.relu(acc_conv1)
		acc_conv1_shape = acc_conv1.get_shape().as_list()
		acc_conv1, acc_dropB1 = DeepIoT_dropOut.dropout(acc_conv1, dropOut_prun(prob_list_acc1, prun_thres, sol_train), is_training=train, 
			noise_shape=[acc_conv1_shape[0], 1, 1, acc_conv1_shape[3]], name='acc_dropout1')
		out_binary_mask[u'acc_conv1'] = acc_dropB1


		acc_conv2 = layers.convolution2d(acc_conv1, CONV_NUM, kernel_size=[1, CONV_LEN_INTE],
						stride=[1, 1], padding='VALID', activation_fn=None, data_format='NHWC', scope='acc_conv2')
		acc_conv2 = batch_norm_layer(acc_conv2, train, scope='acc_BN2')
		acc_conv2 = tf.nn.relu(acc_conv2)
		acc_conv2_shape = acc_conv2.get_shape().as_list()
		acc_conv2, acc_dropB2 = DeepIoT_dropOut.dropout(acc_conv2, dropOut_prun(prob_list_acc2, prun_thres, sol_train), is_training=train,
			noise_shape=[acc_conv2_shape[0], 1, 1, acc_conv2_shape[3]], name='acc_dropout2')
		out_binary_mask[u'acc_conv2'] = acc_dropB2


		acc_conv3 = layers.convolution2d(acc_conv2, CONV_NUM, kernel_size=[1, CONV_LEN_LAST],
						stride=[1, 1], padding='VALID', activation_fn=None, data_format='NHWC', scope='acc_conv3')
		acc_conv3 = batch_norm_layer(acc_conv3, train, scope='acc_BN3')
		acc_conv3 = tf.nn.relu(acc_conv3)
		acc_conv3_shape = acc_conv3.get_shape().as_list()
		acc_conv_out = tf.reshape(acc_conv3, [acc_conv3_shape[0], acc_conv3_shape[1], 1, acc_conv3_shape[2],acc_conv3_shape[3]])


		gyro_conv1 = layers.convolution2d(gyro_inputs, CONV_NUM, kernel_size=[1, 2*3*CONV_LEN],
						stride=[1, 2*3], padding='VALID', activation_fn=None, data_format='NHWC', scope='gyro_conv1')
		gyro_conv1 = batch_norm_layer(gyro_conv1, train, scope='gyro_BN1')
		gyro_conv1 = tf.nn.relu(gyro_conv1)
		gyro_conv1_shape = gyro_conv1.get_shape().as_list()
		gyro_conv1, gyro_dropB1 = DeepIoT_dropOut.dropout(gyro_conv1, dropOut_prun(prob_list_gyro1, prun_thres, sol_train), is_training=train,
			noise_shape=[gyro_conv1_shape[0], 1, 1, gyro_conv1_shape[3]], name='gyro_dropout1')
		out_binary_mask[u'gyro_conv1'] = gyro_dropB1


		gyro_conv2 = layers.convolution2d(gyro_conv1, CONV_NUM, kernel_size=[1, CONV_LEN_INTE],
						stride=[1, 1], padding='VALID', activation_fn=None, data_format='NHWC', scope='gyro_conv2')
		gyro_conv2 = batch_norm_layer(gyro_conv2, train, scope='gyro_BN2')
		gyro_conv2 = tf.nn.relu(gyro_conv2)
		gyro_conv2_shape = gyro_conv2.get_shape().as_list()
		gyro_conv2, gyro_dropB2 = DeepIoT_dropOut.dropout(gyro_conv2, dropOut_prun(prob_list_gyro2, prun_thres, sol_train), is_training=train,
			noise_shape=[gyro_conv2_shape[0], 1, 1, gyro_conv2_shape[3]], name='gyro_dropout2')
		out_binary_mask[u'gyro_conv2'] = gyro_dropB2


		gyro_conv3 = layers.convolution2d(gyro_conv2, CONV_NUM, activation_fn=None, kernel_size=[1, CONV_LEN_LAST],
						stride=[1, 1], padding='VALID', data_format='NHWC', scope='gyro_conv3')
		gyro_conv3 = batch_norm_layer(gyro_conv3, train, scope='gyro_BN3')
		gyro_conv3 = tf.nn.relu(gyro_conv3)
		gyro_conv3_shape = gyro_conv3.get_shape().as_list()
		gyro_conv_out = tf.reshape(gyro_conv3, [gyro_conv3_shape[0], gyro_conv3_shape[1], 1, gyro_conv3_shape[2], gyro_conv3_shape[3]])	


		sensor_conv_in = tf.concat([acc_conv_out, gyro_conv_out], 2)
		senor_conv_shape = sensor_conv_in.get_shape().as_list()
		sensor_conv_in, sensor_in_dropB = DeepIoT_dropOut.dropout(sensor_conv_in, dropOut_prun(prob_list_senIn, prun_thres, sol_train), is_training=train,
			noise_shape=[senor_conv_shape[0], 1, 1, 1, senor_conv_shape[4]], name='sensor_dropout_in')
		out_binary_mask[u'gyro_conv3'] = sensor_in_dropB
		out_binary_mask[u'acc_conv3'] = sensor_in_dropB


		sensor_conv1 = layers.convolution2d(sensor_conv_in, CONV_NUM2, kernel_size=[1, 2, CONV_MERGE_LEN],
						stride=[1, 1, 1], padding='SAME', activation_fn=None, data_format='NDHWC', scope='sensor_conv1')
		sensor_conv1 = batch_norm_layer(sensor_conv1, train, scope='sensor_BN1')
		sensor_conv1 = tf.nn.relu(sensor_conv1)
		sensor_conv1_shape = sensor_conv1.get_shape().as_list()
		sensor_conv1, sensor_dropB1 = DeepIoT_dropOut.dropout(sensor_conv1, dropOut_prun(prob_list_sen1, prun_thres, sol_train), is_training=train,
			noise_shape=[sensor_conv1_shape[0], 1, 1, 1, sensor_conv1_shape[4]], name='sensor_dropout1')
		out_binary_mask[u'sensor_conv1'] = sensor_dropB1
		

		sensor_conv2 = layers.convolution2d(sensor_conv1, CONV_NUM2, kernel_size=[1, 2, CONV_MERGE_LEN2],
						stride=[1, 1, 1], padding='SAME', activation_fn=None, data_format='NDHWC', scope='sensor_conv2')
		sensor_conv2 = batch_norm_layer(sensor_conv2, train, scope='sensor_BN2')
		sensor_conv2 = tf.nn.relu(sensor_conv2)
		sensor_conv2_shape = sensor_conv2.get_shape().as_list()
		sensor_conv2, sensor_dropB2 = DeepIoT_dropOut.dropout(sensor_conv2, dropOut_prun(prob_list_sen2, prun_thres, sol_train), is_training=train, 
			noise_shape=[sensor_conv2_shape[0], 1, 1, 1, sensor_conv2_shape[4]], name='sensor_dropout2')
		out_binary_mask[u'sensor_conv2'] = sensor_dropB2
		

		sensor_conv3 = layers.convolution2d(sensor_conv2, CONV_NUM2, kernel_size=[1, 2, CONV_MERGE_LEN3],
						stride=[1, 1, 1], padding='SAME', activation_fn=None, data_format='NDHWC', scope='sensor_conv3')
		sensor_conv3 = batch_norm_layer(sensor_conv3, train, scope='sensor_BN3')
		sensor_conv3 = tf.nn.relu(sensor_conv3)
		sensor_conv3_shape = sensor_conv3.get_shape().as_list()
		sensor_conv3, sensor_dropB3 = DeepIoT_dropOut.dropout(sensor_conv3, dropOut_prun(prob_list_sen3, prun_thres, sol_train), is_training=train, 
			noise_shape=[sensor_conv3_shape[0], 1, 1, 1, sensor_conv3_shape[4]], name='sensor_dropout3')
		out_binary_mask[u'sensor_conv3'] = sensor_dropB3

		sensor_conv_out = tf.reshape(sensor_conv3, [sensor_conv3_shape[0], sensor_conv3_shape[1], sensor_conv3_shape[2]*sensor_conv3_shape[3]*sensor_conv3_shape[4]])

		gru_cell1 = tf.contrib.rnn.GRUCell(INTER_DIM)
		gru_bMask1 = DeepIoT_rnnDrop.GenRNNMask(keep_prob=dropOut_prun(prob_list_rnn1, prun_thres, sol_train), is_training=train, batch_size=BATCH_SIZE, inter_dim=INTER_DIM)
		out_binary_mask[u'cell_0'] = gru_bMask1
		gru_cell1 = DeepIoT_rnnDrop.DropoutWrapper(gru_cell1, binary_tensor=gru_bMask1)
		

		gru_cell2 = tf.contrib.rnn.GRUCell(INTER_DIM)
		gru_bMask2 = DeepIoT_rnnDrop.GenRNNMask(keep_prob=dropOut_prun(prob_list_rnn2, prun_thres, sol_train), is_training=train, batch_size=BATCH_SIZE, inter_dim=INTER_DIM)
		gru_cell2 = DeepIoT_rnnDrop.DropoutWrapper(gru_cell2, binary_tensor=gru_bMask2)
		out_binary_mask[u'cell_1'] = gru_bMask2


		cell = tf.contrib.rnn.MultiRNNCell([gru_cell1, gru_cell2])
		init_state = cell.zero_state(BATCH_SIZE, tf.float32)

		cell_output, final_stateTuple = tf.nn.dynamic_rnn(cell, sensor_conv_out, sequence_length=length, initial_state=init_state, time_major=False)

		sum_cell_out = tf.reduce_sum(cell_output*mask, axis=1, keep_dims=False)
		avg_cell_out = sum_cell_out/avgNum

		logits = layers.fully_connected(avg_cell_out, OUT_DIM, activation_fn=None, scope='output')

		return logits, out_binary_mask
###### DeepSense Part End

###### Compressor Part Start
def concat_weight_mat(weight_dict):
	cancat_weight_dict = {}
	for layer_name in weight_dict.keys():
		if not 'cell' in layer_name:
			cur_w = weight_dict[layer_name][u'weights:0']
			cur_b = weight_dict[layer_name][u'biases:0']
			cur_w_shape = cur_w.get_shape().as_list()
			new_w_shape_a = 1
			for idx in xrange(len(cur_w_shape)-1):
				new_w_shape_a *= cur_w_shape[idx]
			new_w_shape_b = cur_w_shape[-1]
			cur_w = tf.reshape(cur_w, [new_w_shape_a, new_w_shape_b])
			cur_b = tf.expand_dims(cur_b, 0)
			weight = tf.concat(axis=0, values=[cur_w, cur_b])
			cancat_weight_dict[layer_name] = weight
		else:
			gates_w = weight_dict[layer_name][u'gates'][u'kernel:0']
			gates_b = weight_dict[layer_name][u'gates'][u'bias:0']
			candidate_w = weight_dict[layer_name][u'candidate'][u'kernel:0']
			candidate_b = weight_dict[layer_name][u'candidate'][u'bias:0']

			gates_b = tf.expand_dims(gates_b, 0)
			gates_weight_pre = 	tf.concat(axis=0, values=[gates_w, gates_b])
			gates_weight_0, gates_weight_1 = tf.split(axis=1, num_or_size_splits=2, value=gates_weight_pre)
			gates_weight = tf.concat(axis=0, values=[gates_weight_0, gates_weight_1])

			candidate_b = tf.expand_dims(candidate_b, 0)
			candidate_weight = tf.concat(axis=0, values=[candidate_w, candidate_b])

			weight = tf.concat(axis=0, values=[gates_weight, candidate_weight])
			cancat_weight_dict[layer_name] = weight

	return cancat_weight_dict

def merg_ord_mat(cancat_weight_dict, ord_list):
	weight_list = []
	for ord_elem in ord_list:
		if type(ord_elem) == type([]):
			sub_weights = [cancat_weight_dict[sub_ord_elem] for sub_ord_elem in ord_elem]
			if '3' in ord_elem[0]:
				weight = tf.concat(axis=0, values=sub_weights)
			else:
				weight = tf.concat(axis=1, values=sub_weights)
		else:
			weight = cancat_weight_dict[ord_elem] 
		weight_list.append(weight)
	return weight_list

def trans_mat2vec(weight_list, vec_dim):
	vec_list = []
	for idx, weight in enumerate(weight_list):
		weight_shape = weight.get_shape().as_list()
		matrix1 = tf.get_variable("trans_W"+str(idx)+"a", [1, weight_shape[0]], tf.float32,
			tf.truncated_normal_initializer(stddev=np.power(2.0/(1 + weight_shape[0]),0.5)))
		matrix2 = tf.get_variable("trans_W"+str(idx)+"b", [weight_shape[1], vec_dim], tf.float32,
			tf.truncated_normal_initializer(stddev=np.power(2.0/(weight_shape[1] + vec_dim),0.5)))
		vec = tf.squeeze(tf.matmul(tf.matmul(matrix1, weight), matrix2), [0])
		vec_list.append(vec)
	return vec_list

def transback(out_list, weight_list, ord_list):
	drop_prob_dict = {}
	for idx in xrange(len(out_list)):
		cell_out = out_list[idx]
		weight = weight_list[idx]
		layer_name = ord_list[idx]
		cell_out_shape = cell_out.get_shape().as_list()
		weight_shape = weight.get_shape().as_list()
		maxtrix = tf.get_variable("transBack_W"+str(idx), [cell_out_shape[1], weight_shape[1]], tf.float32,
						tf.truncated_normal_initializer(stddev=np.power(2.0/(cell_out_shape[1] + weight_shape[1]),0.5)))
		drop_out_prob = tf.nn.sigmoid(tf.matmul(cell_out, maxtrix))
		if type(layer_name) == type([]):
			if '3' in layer_name[0]:
				for sub_layer_name in layer_name:
					drop_prob_dict[sub_layer_name] = drop_out_prob
			else:
				drop_out_prob0, drop_out_prob1 = tf.split(axis=1, num_or_size_splits=2, value=drop_out_prob)
				drop_prob_dict[layer_name[0]] = drop_out_prob0
				drop_prob_dict[layer_name[1]] = drop_out_prob1
		else:
			drop_prob_dict[layer_name] = drop_out_prob
	return drop_prob_dict


def compressor(d_vars, inter_dim = 64, reuse=False, name='compressor'):
	with tf.variable_scope(name, reuse=reuse) as scope:
		org_weight_dict = {}
		for var in d_vars:
			if '_BN' in var.name:
				continue
			if not 'deepSense/' in var.name:
				continue
			var_name_list = var.name.split('/')
			if len(var_name_list) == 3:
				if not var_name_list[1] in org_weight_dict.keys():
					org_weight_dict[var_name_list[1]] = {}
				org_weight_dict[var_name_list[1]][var_name_list[2]] = var 
			elif len(var_name_list) == 7:
				if not var_name_list[3] in org_weight_dict.keys():
					org_weight_dict[var_name_list[3]] = {}
				if not var_name_list[5] in org_weight_dict[var_name_list[3]].keys():
					org_weight_dict[var_name_list[3]][var_name_list[5]] = {}
				org_weight_dict[var_name_list[3]][var_name_list[5]][var_name_list[6]] = var

		cancat_weight_dict = concat_weight_mat(org_weight_dict)
		# ord_list = [[u'acc_conv1', u'gyro_conv1'], [u'acc_conv2', u'gyro_conv2'], [u'acc_conv3', u'gyro_conv3'], 
		# 			u'sensor_conv1', u'sensor_conv2', u'sensor_conv3', u'cell_0', u'cell_1', u'output']
		ord_list = [[u'acc_conv1', u'gyro_conv1'], [u'acc_conv2', u'gyro_conv2'], [u'acc_conv3', u'gyro_conv3'], 
					u'sensor_conv1', u'sensor_conv2', u'sensor_conv3', u'cell_0', u'cell_1']
		weight_list = merg_ord_mat(cancat_weight_dict, ord_list)

		vec_list = trans_mat2vec(weight_list, inter_dim)
		vec_input = tf.stack(vec_list)
		vec_input = tf.expand_dims(vec_input, 0)

		lstm_cell = tf.contrib.rnn.LSTMCell(inter_dim)
		init_state = lstm_cell.zero_state(1, tf.float32)
		cell_output, final_stateTuple = tf.nn.dynamic_rnn(lstm_cell, vec_input, initial_state=init_state, time_major=False)
		cell_output = tf.squeeze(cell_output, [0])
		cell_output_list = tf.split(axis=0, num_or_size_splits=len(ord_list), value=cell_output)

		drop_prob_dict = transback(cell_output_list, weight_list, ord_list)

	return drop_prob_dict

def gen_compressor_loss(drop_prob_dict, out_binary_mask, batchLoss, ema, lossMean, lossStd):
	compsBatchLoss = 0
	for layer_name in drop_prob_dict.keys():
		drop_prob = dropOut_prun(drop_prob_dict[layer_name], prun_thres, sol_train)
		out_binary = out_binary_mask[layer_name]
		if 'conv' in layer_name:
			out_binary = tf.squeeze(out_binary)
		drop_prob  = tf.tile(drop_prob, [BATCH_SIZE, 1])
		neg_drop_prob = 1.0 - drop_prob
		neg_out_binary = tf.abs(1.0 - out_binary)
		compsBatchLoss += tf.reduce_sum(tf.log(drop_prob*out_binary + neg_drop_prob*neg_out_binary), 1)
	compsBatchLoss *= (batchLoss - ema.average(lossMean))/tf.maximum(1.0, ema.average(lossStd))
	compsLoss = tf.reduce_mean(compsBatchLoss)
	return compsLoss

def update_drop_op(drop_prob_dict, prob_list_dict):
	update_drop_op_dict = {}
	for layer_name in drop_prob_dict.keys():
		prob_list = prob_list_dict[layer_name]
		prob_list_shape = prob_list.get_shape().as_list()
		drop_prob = drop_prob_dict[layer_name]
		update_drop_op_dict[layer_name] = tf.assign(prob_list, tf.reshape(drop_prob, prob_list_shape))
	return update_drop_op_dict
###### Compressor Part End


global_step = tf.Variable(0, trainable=False)
comps_global_step = tf.Variable(0, trainable=False)
prun_global_step = tf.Variable(0, trainable=False)
sol_train_global_step = tf.Variable(0, trainable=False)

TF_RECORD_PATH = 'sepHARData_'+select
SAVER_DIR = 'deepHHAR_saver'

batch_feature, batch_label = input_pipeline_har(os.path.join(TF_RECORD_PATH, 'train.tfrecord'), BATCH_SIZE)
batch_eval_feature, batch_eval_label = input_pipeline_har(os.path.join(TF_RECORD_PATH, 'eval.tfrecord'), BATCH_SIZE, shuffle_sample=False)

###### DeepSense Model Start
logits, out_binary_mask = deepSense(batch_feature, True, name='deepSense')

predict = tf.argmax(logits, axis=1)

batchLoss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=batch_label)
batchLossMean, batchLossVar = tf.nn.moments(batchLoss, axes = [0])
lossMean = tf.reduce_mean(batchLossMean)
lossStd = tf.reduce_mean(tf.sqrt(batchLossVar))
loss = tf.reduce_mean(batchLoss)

logits_eval, out_binary_mask_eval = deepSense(batch_eval_feature, False, reuse=True, name='deepSense')
predict_eval = tf.argmax(logits_eval, axis=1)
loss_eval = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits_eval, labels=batch_eval_label))

t_vars = tf.trainable_variables()
d_vars = [var for var in t_vars if 'deepSense/' in var.name]

regularizers = 0.
for var in d_vars:
	regularizers += tf.nn.l2_loss(var)
loss += 5e-4 * regularizers

discOptimizer = tf.train.AdamOptimizer(
		learning_rate=1e-4, 
		beta1=0.5,
		beta2=0.9
	).minimize(loss, var_list=d_vars)
###### DeepSense Model End

movingAvg_decay = 0.99
ema = tf.train.ExponentialMovingAverage(0.9)
maintain_averages_op = ema.apply([lossMean, lossStd])
movingAvg_batchLoss = tf.get_variable("movingAvg_batchLoss", [], tf.float32, tf.constant_initializer(0.0), trainable=False)
movingStd_batchLoss = tf.get_variable("movingStd_batchLoss", [], tf.float32, tf.constant_initializer(1.0), trainable=False)

drop_prob_dict = compressor(d_vars)

compsLoss = gen_compressor_loss(drop_prob_dict, out_binary_mask, batchLoss, ema, lossMean, lossStd)
update_drop_op_dict = update_drop_op(drop_prob_dict, prob_list_dict)

t_vars = tf.trainable_variables()
no_c_vars = [var for var in t_vars if not 'compressor/' in var.name]
c_vars = [var for var in t_vars if 'compressor/' in var.name]

compsOptimizer = tf.train.RMSPropOptimizer(0.001).minimize(compsLoss,
		var_list=c_vars, global_step=comps_global_step)

left_num_dict = count_prun(prob_list_dict, prun_thres)

START_THRES = 0.0
FINAL_THRES = 0.825
THRES_STEP = 33
UPDATE_STEP = 500

with tf.Session() as sess:
	tf.global_variables_initializer().run()
	saver = tf.train.Saver()
	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(coord=coord)

	###### Start Load Test DeepSense Pre-Trained Model
	print 'Loading Pre-trained Uncompressed DeeepSense Model'
	saver.restore(sess, os.path.join(SAVER_DIR, "model.ckpt"))
	print 'Loaded\n'

	dev_accuracy = []
	dev_cross_entropy = []
	for eval_idx in xrange(EVAL_ITER_NUM):
		eval_loss_v, _trainY, _predict = sess.run([loss_eval, batch_eval_label, predict_eval])

		_label = np.argmax(_trainY, axis=1)
		_accuracy = np.mean(_label == _predict)
		dev_accuracy.append(_accuracy)
		dev_cross_entropy.append(eval_loss_v)
	plot.plot('dev accuracy', np.mean(dev_accuracy))
	plot.plot('dev cross entropy', np.mean(dev_cross_entropy))
	print 'Uncompressed DeepSense Model'
	print 'Dev accuracy', np.mean(dev_accuracy)
	print 'Dev cross entropy', np.mean(dev_cross_entropy)
	print '\n'
	plot.tick()
	###### End Load Test DeepSense Pre-Trained Model

	
	###### Start Compressing Part
	thres_update_count = 0
	sess.run(tf.assign(sol_train, 0.0))
	for iteration in xrange(TOTAL_ITER_NUM):

		# Train Critic
		_, lossV, _trainY, _predict = sess.run([discOptimizer, loss, batch_label, predict])

		# Train Compressor
		_, compsLossV, _, lossMeanV, lossStdV = sess.run([compsOptimizer, compsLoss, maintain_averages_op, 
																ema.average(lossMean), ema.average(lossStd)])
		_label = np.argmax(_trainY, axis=1)
		_accuracy = np.mean(_label == _predict)
		plot.plot('train cross entropy', lossV)
		plot.plot('train accuracy', _accuracy)
		plot.plot('train comps loss', compsLossV)

		for layer_name in update_drop_op_dict.keys():
			sess.run(update_drop_op_dict[layer_name])

		if iteration % UPDATE_STEP == 0 and thres_update_count <= THRES_STEP:
			cur_thres = START_THRES + thres_update_count*(FINAL_THRES - START_THRES)/THRES_STEP
			print 'Cur Threshold:', cur_thres
			sess.run(tf.assign(prun_thres, cur_thres))
			thres_update_count += 1

		if iteration % 200 == 199:
			dev_accuracy = []
			dev_cross_entropy = []
			for eval_idx in xrange(EVAL_ITER_NUM):
				eval_loss_v, _trainY, _predict = sess.run([loss_eval, batch_eval_label, predict_eval])
				_label = np.argmax(_trainY, axis=1)
				_accuracy = np.mean(_label == _predict)
				dev_accuracy.append(_accuracy)
				dev_cross_entropy.append(eval_loss_v)
			plot.plot('dev accuracy', np.mean(dev_accuracy))
			plot.plot('dev cross entropy', np.mean(dev_cross_entropy))
			cur_left_num = gen_cur_prun(sess, left_num_dict)
			print 'Left Element in DeepSense:', cur_left_num
			cur_comps_ratio = compress_ratio(cur_left_num, org_dim_dict)
			if cur_comps_ratio < 7.0 and np.mean(dev_accuracy) >= 0.93:
				break

		if (iteration < 5) or (iteration % 200 == 199):
			plot.flush()

		plot.tick()
	###### End Init Compressor Part

	###### Start Fine-Tune DeepSense Part
	print '\nStart Fine-tunning'
	sess.run(tf.assign(sol_train, 1.0))
	cur_left_num = gen_cur_prun(sess, left_num_dict)
	print 'Compressed DeepSense Model:', cur_left_num
	compress_ratio(cur_left_num, org_dim_dict)
	for iteration in xrange(TOTAL_ITER_NUM):
		_, lossV, _trainY, _predict = sess.run([discOptimizer, loss, batch_label, predict])
		_label = np.argmax(_trainY, axis=1)
		_accuracy = np.mean(_label == _predict)
		plot.plot('train cross entropy', lossV)
		plot.plot('train accuracy', _accuracy)


		if iteration % 50 == 49:
			dev_accuracy = []
			dev_cross_entropy = []
			for eval_idx in xrange(EVAL_ITER_NUM):
				eval_loss_v, _trainY, _predict = sess.run([loss_eval, batch_eval_label, predict_eval])
				_label = np.argmax(_trainY, axis=1)
				_accuracy = np.mean(_label == _predict)
				dev_accuracy.append(_accuracy)
				dev_cross_entropy.append(eval_loss_v)
			plot.plot('dev accuracy', np.mean(dev_accuracy))
			plot.plot('dev cross entropy', np.mean(dev_cross_entropy))

		if (iteration < 5) or (iteration % 50 == 49):
			plot.flush()

		plot.tick()
	###### End Fine-Tune DeepSense Part


