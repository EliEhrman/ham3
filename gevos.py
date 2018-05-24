from __future__ import print_function
import random
import csv
import os
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import tensorflow as tf

FLAGS = tf.flags.FLAGS

num_words = 100 # 10000 # 400000
c_rsize = 137
c_small_rsize = 99
c_b_learn_hd = True

glove_fn = '../../data/glove/glove.6B.50d.txt'
# c_num_gloves = 10
g_word_vec_len = -1
if c_b_learn_hd:
	tf.flags.DEFINE_float('nn_lrn_rate', 100.0,
						  'base learning rate for nn ')
	c_key_dim = 128
else:
	tf.flags.DEFINE_float('nn_lrn_rate', 0.001,
						  'base learning rate for nn ')
	c_key_dim = 32
c_gg_num_learn_steps = 50000
c_gg_learn_test_every = 100
c_eval_every = 1000
c_sigmoid_factor = 1.0
c_devi_factor = 0.1
c_num_testers = 100
c_num_cd_winners = 10
c_num_ham_winners = 5
c_num_Ws = 10 # 100
c_mutate_factor = 10.0
c_param_init_base = 100.0
c_num_iters = 1000
c_b_test_recall = True

if c_b_test_recall:
	c_num_cd_winners = 1 # r X @ ..
	c_num_ham_winners = 10 # r .. @ X

# FLAGS.nn_lrn_rate = 0.01

def  cdcomp(p1, p2):
	if p1[0] > p2[0]:
		return 1
	else:
		return -1

def cdkey(p):
	return p[0]

x = np.array([9, 4, 4, 3, 3, 9, 0, 4, 6, 0])
ind = np.argpartition(x, -4)[-4:]

rtesters = random.sample(xrange(num_words), c_num_testers)

def eval_progress(word_arr, bin_arr):
	global rtesters
	num_hits, num_poss = 0.0, 0.0
	for tester in rtesters:
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
			# hd_cands = sorted(hd, key=lambda x: x[0], reverse=False)[1:c_num_ham_winners + 1]
		else:
			hd = [[np.dot(test_vec, one_vec), ione] for ione, one_vec in enumerate(bin_arr)]
			hd_cands = sorted(hd, key=lambda x: x[0], reverse=True)[1:c_num_ham_winners+1]
			hd_winners = [cand[1] for cand in hd_cands]
		if c_b_test_recall:
			cd_in_hd = [iword for iword in cd_winners if iword in hd_winners]
			num_hits += float(len(cd_in_hd))
			num_poss += float(c_num_cd_winners)
		else:
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
	return word_dict, np.array(word_arr)


word_dict, word_arr = load_word_dict()
arr_median = np.tile(np.median(word_arr, axis=1), (word_arr.shape[1], 1)).transpose()
arr_bin = np.greater(word_arr, arr_median).astype(np.float32)
baseline_score = eval_progress(word_arr, arr_bin)
# ph_words = tf.placeholder(tf.int32, shape=(), name='ph_words')
tfc_words = tf.constant(word_arr)
ph_o = tf.placeholder(tf.float32, shape=([None]), name='ph_o')

v_r1 = tf.Variable(tf.zeros([c_rsize], dtype=tf.int32),
				   trainable=False, name='v_r1')
v_r2 = tf.Variable(tf.zeros([c_rsize], dtype=tf.int32),
				   trainable=False, name='v_r2')
op_r1 = tf.assign(v_r1, tf.random_uniform([c_rsize], minval=0, maxval=num_words, dtype=tf.int32),
				  name='op_r1')
op_r2 = tf.assign(v_r2, tf.random_uniform([c_rsize], minval=0, maxval=num_words, dtype=tf.int32),
				  name='op_r2')

v_smallr1 = tf.Variable(tf.zeros([c_small_rsize], dtype=tf.int32),
				   trainable=False, name='v_r1')
v_smallr2 = tf.Variable(tf.zeros([c_small_rsize], dtype=tf.int32),
				   trainable=False, name='v_r2')

weight_factor = 100.0 / (g_word_vec_len * c_key_dim)
# weight_factor = 1.0 / 150 * 15
# t_shape = tf.constant([2], dtype=tf.int32)
var_scope = 'main_scope'
v_W = []
t_diff = []
with tf.variable_scope(var_scope, reuse=None):
	# v_W = tf.Variable(tf.random_normal(shape=[num_inputs, c_key_dim], mean=0.0, stddev=weight_factor), dtype=tf.float32)
	# for ibit in range(c_key_dim):
	# v_Wpre = tf.get_variable('v_Wpre', shape=[g_word_vec_len, g_word_vec_len], dtype=tf.float32,
	# 					  initializer=tf.random_normal_initializer(mean=0.0, stddev=weight_factor))
	for iW in range(c_num_Ws):
		v_W.append(tf.get_variable('v_W'+str(iW), shape=[g_word_vec_len, c_key_dim], dtype=tf.float32,
							  initializer=tf.random_normal_initializer(mean=0.0, stddev=weight_factor)))


with tf.name_scope(var_scope):
	# ph_input = tf.placeholder(tf.float32, shape=(None, g_word_vec_len), name='input')

	# t_y = []
	# for ibit in range(c_key_dim):
	# t_t = tf.nn.relu(tf.matmul(tfc_words, v_Wpre), name='t_t')
	# t_y = tf.sigmoid(tf.multiply(tf.matmul(t_t, v_W), c_sigmoid_factor), name='t_y_')
	t_y, t_bin = [], []

	for iW in range(c_num_Ws):
		if c_b_learn_hd:
			one_y = tf.sigmoid(tf.multiply(tf.matmul(tfc_words, v_W[iW]), c_sigmoid_factor), name='t_y_')
			# one_y = tf.multiply(tf.matmul(tfc_words, v_W[iW]), c_sigmoid_factor)
			t_y.append(tf.where(tf.greater(one_y, 0.50), tf.ones_like(one_y), tf.zeros_like(one_y)))
		else:
			one_y = tf.nn.l2_normalize(tf.multiply(tf.matmul(tfc_words, v_W[iW]), c_sigmoid_factor), dim=1)
			t_y.append(one_y)

		# t_bin.append(tf.where(tf.greater(t_y[0], 0.50), tf.ones_like(t_y[0]), tf.zeros_like(t_y[0])))

for iW in range(c_num_Ws):
	t_o1 = tf.gather(tfc_words, v_r1, name='t_o1')
	t_o2 = tf.gather(tfc_words, v_r2, name='t_o2')

	t_y1 = tf.gather(t_y[iW], v_r1, name='t_y1')
	t_y2 = tf.gather(t_y[iW], v_r2, name='t_y2')

	t_so1 = tf.gather(tfc_words, v_smallr1, name='t_o1')
	t_so2 = tf.gather(tfc_words, v_smallr2, name='t_o2')

	t_sy1 = tf.gather(t_y[iW], v_smallr1, name='t_y1')
	t_sy2 = tf.gather(t_y[iW], v_smallr2, name='t_y2')


# t_cdo = tf.where(tf.equal(t_o1, t_o2), tf.ones([c_rsize], dtype=tf.float32),
# 				 tf.zeros([c_rsize], dtype=tf.float32), name='t_cdo')
	t_cdo = tf.reduce_sum(tf.multiply(t_o1, t_o2), axis=1, name='t_cdo')
	t_cdso = tf.reduce_sum(tf.multiply(t_so1, t_so2), axis=1, name='t_cdso')
	if c_b_learn_hd:
		hamm = tf.subtract(1.0, tf.abs(tf.subtract(t_y1, t_y2)))
		t_cdy = tf.divide(tf.reduce_sum(hamm, axis=1, name='t_cdy'), float(c_key_dim))
		# shamm = tf.subtract(1.0, tf.abs(tf.subtract(t_sy1, t_sy2)))
		# t_cdsy = tf.divide(tf.reduce_sum(shamm, axis=1, name='t_cdsy'), float(c_key_dim))
	else:
		t_cdy = tf.reduce_sum(tf.multiply(t_y1, t_y2), axis=1, name='t_cdy')
		# t_cdsy = tf.reduce_sum(tf.multiply(t_sy1, t_sy2), axis=1, name='t_cdsy')

	t_diff.append(tf.reduce_mean((t_cdo - t_cdy) ** 2, name='t_diff'))

t_top_cds, t_top_idxs = tf.nn.top_k(t_cdy, k = c_small_rsize, sorted=False)
op_sr1 = tf.assign(v_smallr1, tf.gather(v_r1, t_top_idxs))
op_sr2 = tf.assign(v_smallr2, tf.gather(v_r2, t_top_idxs))

t_Ws = tf.transpose(tf.stack(v_W), [1, 2, 0])
ph_wWs = tf.placeholder(dtype=tf.float32, shape=(c_num_Ws))
t_Ws_multiplied = tf.transpose(tf.multiply(t_Ws, ph_wWs), [2, 0, 1])
t_Ws_sum = tf.reduce_sum(ph_wWs)
t_W_meaned = tf.divide(tf.reduce_sum(t_Ws_multiplied, axis=0), t_Ws_sum)
t_Ws_tiled = tf.tile(tf.expand_dims(t_W_meaned, axis=0), [c_num_Ws, 1, 1])
t_W_rand = tf.random_normal([c_num_Ws, g_word_vec_len, c_key_dim], mean=0.0, stddev=weight_factor/c_mutate_factor)
t_mutate = t_Ws_tiled + t_W_rand
t_W = tf.unstack(t_mutate, axis=0)
op_back_to_v_W = [tf.assign(v_W[iw], t_W[iw]) for iw in range(c_num_Ws)]
# t_sdiff = tf.reduce_mean((t_cdso - t_cdsy) ** 2, name='t_sdiff')
# t_devi = tf.reduce_mean((t_bin[0] - t_y[0]) ** 2, name='t_devi')
# if c_b_learn_hd:
# 	t_err = t_sdiff + (c_devi_factor * t_devi)
# else:
# 	t_err = t_sdiff
# op_train_step = tf.train.GradientDescentOptimizer(FLAGS.nn_lrn_rate).minimize(t_err, name='op_train_step')

losses = []
devis = []
diffs = []

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# for iW in range(c_num_Ws):
# 	r_bin = sess.run(t_bin[iW])
# 	init_score = eval_progress(word_arr, r_bin)
#
# exit()

# print(sess.run(v_W))
losses = [[] for _ in range(c_num_Ws)]
sess.run([op_r1, op_r2])
for step in range(c_gg_num_learn_steps):
	sess.run([op_r1, op_r2])
	# sess.run([op_sr1, op_sr2])
	if step % c_gg_learn_test_every == 0:
		# err =
		err = np.mean([np.mean(losses) for _ in range(c_num_Ws)])
		print('lrn step ', step, 'err', err)
		losses = [[] for _ in range(c_num_Ws)]
	# print(sess.run([t_cdo, t_cdy, t_y, hamm, v_r1, v_r2]))
		# print('t_y', sess.run(t_y))
		# print('t_bin', sess.run(t_bin))

	eval_score = []
	if step % c_eval_every == 0:
		for iW in range(c_num_Ws):
			r_bin = sess.run(t_y[iW])
			eval_score.append(eval_progress(word_arr, r_bin))
		print('eval_score = ', np.mean(eval_score))
	# exit()

	diffs = np.zeros([c_num_Ws], dtype=np.float32)
	for iW in range(c_num_Ws):
		nd_diff = sess.run(t_diff[iW])
		losses[iW].append(nd_diff)
		diffs[iW] = nd_diff

	# print(np.mean(diffs), 'vals:', diffs)
	min, max = diffs.min(), diffs.max()
	diffs = (max - diffs) / (max - min)
	# print(diffs)
	# print(sess.run(v_W))
	for iW in range(c_num_Ws):
		sess.run(op_back_to_v_W[iW], feed_dict={ph_wWs: diffs})
	# print(sess.run(v_W))
	# devis.append(nn_outputs[2])
	# diffs.append(nn_outputs[1])

print('v_W', sess.run(v_W[0]))
print('t_y', sess.run(t_y[0]))
