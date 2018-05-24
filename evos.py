from __future__ import print_function
import random
import csv
import sys
import os
import numpy as np
from scipy.special import expit as sigmoid
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"
# import tensorflow as tf

# FLAGS = tf.flags.FLAGS

num_words = 1000 # 400000
c_rsize = 1377
c_small_rsize = 99
c_b_learn_hd = True

glove_fn = '../../data/glove/glove.6B.50d.txt'
# c_num_gloves = 10
g_word_vec_len = -1
if c_b_learn_hd:
	# tf.flags.DEFINE_float('nn_lrn_rate', 100.0,
	# 					  'base learning rate for nn ')
	c_key_dim = 128
else:
	# tf.flags.DEFINE_float('nn_lrn_rate', 0.001,
	# 					  'base learning rate for nn ')
	c_key_dim = 32
c_gg_num_learn_steps = 50000
c_gg_learn_test_every = 100
c_eval_every = 1000
c_sigmoid_factor = 1.0
c_devi_factor = 0.1
c_num_testers = 40
c_num_cd_winners = 10
c_num_ham_winners = 5
c_num_Ws = 100
c_mutate_factor = 10.0
c_param_init_base = 100.0
c_num_iters = 1000



# FLAGS.nn_lrn_rate = 0.01

def  cdcomp(p1, p2):
	if p1[0] > p2[0]:
		return 1
	else:
		return -1

def cdkey(p):
	return p[0]

# rtesters = random.sample(xrange(num_words), c_num_testers)
rtesters = xrange(c_num_testers)
rtesters2 = random.sample(xrange(num_words), num_words/2)

def eval_progress(word_arr, bin_arr, l_testers = rtesters, b_use_all = False):
	# global rtesters
	num_hits, num_poss = 0.0, 0.0
	if b_use_all:
		l_testers = range(c_num_testers, num_words)
	for tester in l_testers:
		test_vec = word_arr[tester]
		cd = np.dot(word_arr, test_vec)
		cd_winners = np.argpartition(cd, -(c_num_cd_winners+1))[-(c_num_cd_winners+1):]
		cd_winners = np.delete(cd_winners, np.where(cd_winners == tester))
		# cd = [[np.dot(test_vec, one_vec), ione] for ione, one_vec in enumerate(word_arr)]
		# cands = sorted(cd, key=lambda x: x[0], reverse=True)[1:c_num_cd_winners+1]
		# cd_winners = [cand[1] for cand in cands]
		test_vec = bin_arr[tester]
		if c_b_learn_hd:
			hd = [[np.sum(np.absolute(np.subtract(test_vec, one_vec)), axis=0), ione]  for ione, one_vec in enumerate(bin_arr)]
			hd = np.sum(np.absolute(np.subtract(test_vec, bin_arr)), axis=1)
			hd_winners = np.argpartition(hd, (c_num_ham_winners + 1))[:(c_num_ham_winners + 1)]
			hd_winners = np.delete(hd_winners, np.where(hd_winners == tester))
			# hd = [[np.sum(np.absolute(np.subtract(test_vec, one_vec)), axis=0), ione]  for ione, one_vec in enumerate(bin_arr)]
			# hd_cands = sorted(hd, key=lambda x: x[0], reverse=False)[1:c_num_ham_winners + 1]
		else:
			hd = [[np.dot(test_vec, one_vec), ione] for ione, one_vec in enumerate(bin_arr)]
			hd_cands = sorted(hd, key=lambda x: x[0], reverse=True)[1:c_num_ham_winners+1]
			hd_winners = [cand[1] for cand in hd_cands]
		hd_in_cd = [iword for iword in hd_winners if iword in cd_winners]
		num_hits += float(len(hd_in_cd))
		num_poss += float(c_num_ham_winners)
	# print(num_hits, 'out of', num_poss, 'for a score of', num_hits / num_poss)
	return num_hits / num_poss



def load_word_dict():
	global g_word_vec_len
	glove_fh = open(glove_fn, 'rb')
	glove_csvr = csv.reader(glove_fh, delimiter=' ', quoting=csv.QUOTE_NONE)

	word_dict = {}
	word_arr = []
	for irow, row in enumerate(glove_csvr):
		word = row[0]
		vec = [float(val) for val in row[1:]]
		vec = np.array(vec, dtype=np.float32)
		en = np.linalg.norm(vec, axis=0)
		vec = vec / en
		word_dict[word] = vec
		word_arr.append(vec)
		if irow > num_words:
			break
	# print(row)

	glove_fh.close()
	g_word_vec_len = len(word_dict['the'])
	random.shuffle(word_arr)
	return word_dict, np.array(word_arr)


word_dict, word_arr = load_word_dict()
arr_median = np.tile(np.median(word_arr, axis=1), (word_arr.shape[1], 1)).transpose()
arr_bin = np.greater(word_arr, arr_median).astype(np.float32)
baseline_score = eval_progress(word_arr, arr_bin)

weight_factor = c_param_init_base / (g_word_vec_len * c_key_dim)
nd_one_W = np.random.normal(loc=0.0, scale=weight_factor, size=(g_word_vec_len, c_key_dim))
for iiter in range(c_num_iters):
	print('Iteration #', iiter)
	nd_Wx = np.random.normal(loc=0.0, scale=weight_factor/c_mutate_factor, size=(c_num_Ws, g_word_vec_len, c_key_dim))
	if iiter == 0:
		nd_W = np.random.normal(loc=0.0, scale=weight_factor, size=(c_num_Ws, g_word_vec_len, c_key_dim))
	else:
		nd_W = np.repeat(np.expand_dims(nd_one_W, axis=0), c_num_Ws, axis=0) + nd_Wx
	nd_y = np.dot(word_arr, nd_one_W)
	min_score, max_score = sys.float_info.max, -sys.float_info.max
	l_scores = [0.0 for _ in range(c_num_Ws)]
	for iw in range(c_num_Ws):
		vec = np.dot(word_arr, nd_W[iw])
		if c_b_learn_hd:
			vec = sigmoid(vec * c_sigmoid_factor)
			nd_y = np.where(vec > 0.5, 1.0, 0.0)
		else:
			en = np.linalg.norm(vec, axis=0)
			nd_y = vec / en
		score = eval_progress(word_arr, nd_y, rtesters)
		l_scores[iw] = score
		if score > max_score:
			max_score = score
		if score < min_score:
			min_score = score
	range_scores = (max_score - min_score)
	print('avg score:', np.mean(l_scores))
	nd_sum_Ws = np.zeros((g_word_vec_len, c_key_dim))
	sum_scores = 0.0
	for iw in range(c_num_Ws):
		wW = (l_scores[iw] - min_score) / range_scores
		nd_sum_Ws += nd_W[iw] * wW
		sum_scores += wW

	nd_one_W = nd_sum_Ws / sum_scores
	vec_eval = np.dot(word_arr, nd_one_W)
	if c_b_learn_hd:
		vec_eval = sigmoid(vec_eval * c_sigmoid_factor)
		nd_y_eval = np.where(vec_eval > 0.5, 1.0, 0.0)
	else:
		assert False
	print('eval on same:', eval_progress(word_arr, nd_y_eval, rtesters),
		  'and others:', eval_progress(word_arr, nd_y_eval, rtesters, b_use_all=True))

print('done')