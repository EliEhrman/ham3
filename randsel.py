from __future__ import print_function
import random
import csv
import os
import sys
import copy
import numpy as np

num_words = 1000 # 10000 # 400000
c_rsize = 137
c_small_rsize = 99
c_b_learn_hd = True

glove_fn = '../../data/glove/glove.6B.50d.txt'
# c_num_gloves = 10
g_word_vec_len = -1
if c_b_learn_hd:
	c_key_dim = 128
else:
	c_key_dim = 32
c_gg_num_learn_steps = 50000
c_gg_learn_test_every = 100
c_eval_every = 1000
c_sigmoid_factor = 1.0
c_devi_factor = 0.1
c_num_testers = 80
c_num_cd_winners = 10
c_num_ham_winners = 5
c_num_Ws = 20 # 100
c_mutate_factor = 10.0
c_param_init_base = 100.0
c_num_iters = 1000
c_b_test_recall = False
c_num_paired = 3
c_num_incr_muts = 4
c_num_change_muts = 5
c_num_percentile_stops = 11
c_rnd_asex = 0.2
c_rnd_sex = 0.4 # after asex selection

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

# rtesters = random.sample(xrange(num_words), c_num_testers)
rtesters = xrange(c_num_testers)

def create_pairs(word_arr):
	num_words = word_arr.shape[0]
	# nd_r1, nd_r2 = np.array([num_words*c_num_paired*2]), np.array([num_words*c_num_paired*2])
	r1, r2, targets = [], [], []
	for itester, test_vec in enumerate(word_arr):
		if itester >= c_num_testers:
			break
		cd = np.dot(word_arr, test_vec)
		cd_winners = np.argpartition(cd, -(c_num_paired+1))[-(c_num_paired+1):]
		cd_winners = np.delete(cd_winners, np.where(cd_winners == itester))
		cd_losers = np.argpartition(cd, (c_num_paired))[:c_num_paired]
		r2 += cd_winners.tolist() + cd_losers.tolist()
		r1 += [itester] * c_num_paired * 2
		# targets += [0.0] * c_num_paired + [1.0] * c_num_paired
		targets += [True] * c_num_paired + [False] * c_num_paired

	# shuffle_stick = range(num_words*c_num_paired*2)
	# random.shuffle(shuffle_stick)
	# r1 = [r1[ir] for ir in shuffle_stick]
	# r2 = [r2[ir] for ir in shuffle_stick]
	# targets = [targets[ir] for ir in shuffle_stick]

	return r1, r2, targets


def eval_progress(word_arr, bin_arr, b_use_all=False):
	global rtesters
	rtesters = random.sample(xrange(num_words), c_num_testers)
	num_hits, num_poss = 0.0, 0.0
	l_testers = range(c_num_testers, num_words) if b_use_all else rtesters
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
	random.shuffle(word_arr)
	return word_dict, np.array(word_arr)


word_dict, word_arr = load_word_dict()
r1, r2, targets = create_pairs(word_arr)
l_percentiles = []
for decile in range(1, c_num_percentile_stops+1):
	percentile = float(decile) * (100.0 / float(c_num_percentile_stops+1))
	nd_percentile = np.percentile(word_arr, percentile, axis=0)
	l_percentiles.append(nd_percentile)
nd_percentile = np.stack(l_percentiles).transpose()

def bin_from_W(W):
	nd_wi = np.array([wi for wi, _ in W])
	nd_wt = np.array([wt for _, wt in W])
	bin = [1.0 if word_arr[0][wt] >= nd_percentile[wi][wt] else 0.0 for wi, wt in W]
	nd_val = word_arr[0][nd_wi]
	# nd_thresh = np.array([nd_percentile for wt in nd_wt])
	b = np.zeros((c_key_dim, 11), dtype=np.int32)
	b[np.arange(c_key_dim), nd_wt] = 1
	b = b.astype(np.bool)
	nd_thresh = nd_percentile[nd_wi][b]
	# nd_bin = np.where(nd_val >= nd_thresh, np.ones(c_key_dim), np.zeros(c_key_dim))
	nd_bin = np.stack([np.where(one_word_arr[nd_wi] >= nd_thresh, np.ones(c_key_dim), np.zeros(c_key_dim)) for one_word_arr in word_arr])
	return nd_bin

Ws = []
for iw in range(c_num_Ws):
	Ws.append([[random.randint(0, g_word_vec_len - 1), 5] for i in range(c_key_dim)])
l_scores = [0.0 for _ in range(c_num_Ws)]

for i in range(10000):
	min_score, max_score = sys.float_info.max, -sys.float_info.max
	for iw, W in enumerate(Ws):
		nd_bin = bin_from_W(W)
		score = eval_progress(word_arr, nd_bin)
		if score > max_score:
			max_score = score
		if score < min_score:
			min_score = score
		l_scores[iw] = score
	range_scores = (max_score - min_score)
	print('avg score:', np.mean(l_scores), 'list', l_scores)
	l_Ws = np.array([(score - min_score) / range_scores for score in l_scores])
	sel_prob = l_Ws/np.sum(l_Ws)

	iWs = np.random.choice(c_num_Ws, size=c_num_Ws, p=sel_prob)
	Ws = [copy.deepcopy(Ws[isel]) for isel in iWs]
	for iw, W in enumerate(Ws):
		if random.random() < c_rnd_asex:
			for imut in range(c_num_incr_muts):
				allele = random.randint(0, c_key_dim-1)
				if W[allele][1] < c_num_percentile_stops-2:
					W[allele][1] += 1
			for imut in range(c_num_incr_muts):
				allele = random.randint(0, c_key_dim-1)
				if W[allele][1] > 1:
					W[allele][1] -= 1
			for icmut in range(c_num_change_muts):
				allele = random.randint(0, c_key_dim-1)
				W[allele][0] = random.randint(0, g_word_vec_len - 1)
		elif random.random() < c_rnd_sex:
			partner_W = copy.deepcopy(random.choice(Ws)) # not the numpy function
			for allele in range(c_key_dim):
				if random.random() < 0.5:
					W[allele] = list(partner_W[allele])





nd_median = np.median(word_arr, axis=0)

