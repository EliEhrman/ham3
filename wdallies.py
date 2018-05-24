from __future__ import print_function
import random
import sys
from os.path import expanduser
import numpy as np
import csv
from StringIO import StringIO
import copy

c_alliance_stats_fnt = '~/tmp/alliance_stats.txt'
c_alliance_sel_mat_fnt = '~/tmp/sandbox/alliance_sel_mat.txt'

c_num_testers = 151
c_num_k = 60
c_train_limit = 0.6
c_test_limit = 0.7 # keep a gap between train and est because adjacent records are pretty sililar
c_bitvec_size = 128 # 256
c_min_bits = 2
c_max_bits = 11
c_num_iters = 3000
c_num_sel_mats = 30
c_rnd_asex = 0.3
c_rnd_sex = 0.4 # after asex selection
c_num_incr_muts = 12 # 3
c_num_change_muts = 18 # 5
c_change_mut_prob_change_len = 0.3 # 0.3
c_change_mut_num_change = 12 # 1
c_num_record_breakers = 5
c_num_new_sel_mats_each_time = 3
c_mid_score = c_num_sel_mats / 4



fn = expanduser(c_alliance_stats_fnt)
try:
	with open(fn, 'rb') as fh:
		csvr = csv.reader(fh, delimiter='\t', quoting=csv.QUOTE_NONE, escapechar='\\')
		l_bitvecs, l_diffs = [], []
		for irow, row in enumerate(csvr):
			sbitvec, sdiffs = row

			f = StringIO(sdiffs[1:-1])
			scsv = csv.reader(f, delimiter=',')
			l_diffs.append([int(d) for r in scsv for d in r])
			l_bitvecs.append(np.array([1 if c == '1' else 0 for c in sbitvec]))
		fh.close()

except IOError:
	print('Could not open db_alliance file! Exiting.')
	exit(1)

nd_keys = np.stack(l_bitvecs)
nd_diffs = np.stack(l_diffs)
num_input_bits = nd_keys.shape[1]
num_keys = len(nd_keys)
train_limit = int(float(num_keys) * c_train_limit)
test_limit = int(float(num_keys) * c_test_limit)
# rtesters = xrange(c_num_testers)
rtesters = random.sample(xrange(test_limit, num_keys), c_num_testers)
nd_keys_train = nd_keys[:train_limit]
nd_keys_test = nd_keys[test_limit:]

score = 0.0
for itester in rtesters:
	nd_rearr = np.copy(nd_diffs)
	np.random.shuffle(nd_rearr)
	nd_nearest = nd_rearr[:c_num_k]
	nd_est = np.mean(nd_nearest, axis=0)
	score += np.sum(np.abs(nd_est - nd_diffs[itester]))

print('rand score =', score/(float(c_num_testers)))

score = 0.0
for itester in rtesters:
	test_vec = nd_keys[itester]
	hd = np.sum(np.absolute(np.subtract(test_vec, nd_keys_train)), axis=1)
	hd_winners = np.argpartition(hd, (c_num_k + 1))[:(c_num_k + 1)]
	# hd_winners = np.delete(hd_winners, np.where(hd_winners == itester))
	nd_nearest = nd_diffs[hd_winners]
	nd_est = np.mean(nd_nearest, axis=0)
	score += np.sum(np.abs(nd_est - nd_diffs[itester]))

print('hd score =', score/(float(c_num_testers)))

# num_input_bits = ((phrase_len - 1) * c_bitvec_size) + (c_num_replicate_missing * phrase_len)
# for iiter in range(c_num_iters):

def new_random_sel_mat():
	sel_mat = []
	for ibit in range(c_bitvec_size):
		num_bits = random.randint(c_min_bits, c_max_bits)
		l_sels = []
		for isel in range(num_bits):
			l_sels.append(random.randint(0, num_input_bits - 1))
		sel_mat.append([l_sels, random.randint(1, num_bits)])
	return sel_mat

l_sel_mats = []
l_record_sel_mats = []
l_record_scores = []
for isel in range(c_num_sel_mats):
	l_sel_mats.append(new_random_sel_mat())
	# d_phrase_sel_mats[phrase_len] = sel_mat

for iiter in range(c_num_iters):
	l_scores = [0.0 for _ in range(c_num_sel_mats)]
	min_score, max_score = sys.float_info.max, -sys.float_info.max
	for isel in range(c_num_sel_mats):
		sel_mat = l_sel_mats[isel]
		nd_bits = np.zeros((num_keys, c_bitvec_size), dtype=np.int)
		for ikey in range(num_keys):
			for iobit in range(c_bitvec_size):
				sum = 0
				for iibit in sel_mat[iobit][0]:
					sum += nd_keys[ikey][iibit]
				nd_bits[ikey][iobit] = 1 if sum >= sel_mat[iobit][1] else 0

		score = 0.0
		for itester in rtesters:
			test_vec = nd_bits[itester]
			hd = np.sum(np.absolute(np.subtract(test_vec, nd_bits[:train_limit])), axis=1)
			hd_winners = np.argpartition(hd, (c_num_k + 1))[:(c_num_k + 1)]
			# hd_winners = np.delete(hd_winners, np.where(hd_winners == itester))
			nd_nearest = nd_diffs[hd_winners]
			nd_est = np.mean(nd_nearest, axis=0)
			score += np.sum(np.abs(nd_est - nd_diffs[itester]))

		score = score/(float(c_num_testers))
		# print('isel', isel, 'sel score =', score)
		if score > max_score:
			max_score = score
		if score < min_score:
			min_score = score
		l_scores[isel] = score

	# print('iiter', iiter, 'avg score:', np.mean(l_scores)) # , 'list', l_scores)
	print('iiter', iiter, 'avg score:', np.mean(l_scores), 'min score:', np.min(l_scores)) # , 'list', l_scores)
	if l_record_scores == [] or min_score < l_record_scores[0]:
		l_record_scores.insert(0, min_score)
		l_record_sel_mats.insert(0, l_sel_mats[l_scores.index(min_score)])
	else:
		l_sel_mats[l_scores.index(max_score)] = l_record_sel_mats[0]
		l_scores[l_scores.index(max_score)] = l_record_scores[0]
	# mid_score = (max_score + min_score) / 2.0
	mid_score = l_scores[np.array(l_scores).argsort()[c_mid_score]]
	range_scores = (mid_score - min_score)
	l_sel_scores = np.array([(mid_score - score) / range_scores for score in l_scores])
	l_sel_scores = np.where(l_sel_scores > 0.0, l_sel_scores, np.zeros_like(l_sel_scores))
	sel_prob = l_sel_scores / np.sum(l_sel_scores)

	l_sel_sel_mats = np.random.choice(c_num_sel_mats, size=c_num_sel_mats, p=sel_prob)
	l_sel_mats = [copy.deepcopy(l_sel_mats[isel]) for isel in l_sel_sel_mats]
	for isel, sel_mat in enumerate(l_sel_mats):
		if random.random() < c_rnd_asex:
			for imut in range(c_num_incr_muts):
				allele = random.randint(0, c_bitvec_size-1)
				num_bits = len(sel_mat[allele][0])
				if sel_mat[allele][1] < num_bits-2:
					sel_mat[allele][1] += 1
			for imut in range(c_num_incr_muts):
				allele = random.randint(0, c_bitvec_size-1)
				if sel_mat[allele][1] > 1:
					sel_mat[allele][1] -= 1
			for icmut in range(c_num_change_muts):
				allele = random.randint(0, c_bitvec_size-1)
				bit_list = sel_mat[allele][0]
				if random.random() < c_change_mut_prob_change_len:
					if len(bit_list) < c_max_bits:
						bit_list.append(random.randint(0, num_input_bits - 1))
				elif random.random() < c_change_mut_prob_change_len:
					if len(bit_list) > c_min_bits:
						bit_list.pop(random.randrange(len(bit_list)))
						if sel_mat[allele][1] >= len(bit_list) - 1:
							sel_mat[allele][1] -= 1
				else:
					for ichange in range(c_change_mut_num_change):
						bit_list[random.randint(0, len(bit_list)-1)] = random.randint(0, num_input_bits - 1)
		elif random.random() < c_rnd_sex:
			partner_sel_mat = copy.deepcopy(random.choice(l_sel_mats)) # not the numpy function
			for allele in range(c_bitvec_size):
				if random.random() < 0.5:
					sel_mat[allele] = list(partner_sel_mat[allele])

	for isel in range(c_num_new_sel_mats_each_time):
		l_sel_mats[random.randint(0, c_num_sel_mats-1)] = new_random_sel_mat()

	if (iiter % 10) == 0:
		fh = open(expanduser(c_alliance_sel_mat_fnt), 'wb')
		csvr = csv.writer(fh, delimiter='\t', quoting=csv.QUOTE_NONE, escapechar='\\')
		sel_mat = l_record_sel_mats[0]
		for ibit in range(c_bitvec_size):
			csvr.writerow(sel_mat[ibit])
		fh.close()


print('done')







