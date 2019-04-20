import lightgbm as lgb
import pandas as pd
import gc
from util import *
import os
import sys

a = sys.argv[1]
the_data = sys.argv[2]
params = {
		'boosting_type': 'gbdt',
		'num_leaves': 32,
		# 'reg_alpha':0.0,
		'reg_alpha': 0.0001,
		# 'max_depth' : -1,
		'n_estimators': 1500000,
		'objective': 'binary',
		'subsample': 0.9,
		'colsample_bytree': 0.4,
		'subsample_freq': 1,
		'learning_rate': 0.04,
		'min_child_weight': 50,
		'seed': 11111,
		'metric': 'auc',
		# 'verbose':-1,
		'min_data_in_leaf': 200,
		# 'scale_pos_weight':50,
		'max_bin': 255,
		'two_round':True,
		'histogram_pool_size':131072,
	}



all_feat_list = []

if a == '0':
	# str_decomposition = pd.read_hdf('../cache/str_decomposition.hdf', 'w')
	# str_feat = ['interest1', 'interest2', 'interest5', 'kw1', 'kw2', 'topic1', 'topic2']
	# for each in str_decomposition.columns:
	# 	for fuck in str_feat:
	# 		if fuck in each:
	# 			all_feat_list.append(str_decomposition[[each]])
	
	# all_feat_list.append(pd.read_hdf('../cache/feat5cache.hdf', 'w'))
	# all_feat_list.append(pd.read_hdf('../cache/xin_feat.hdf', 'w'))
	all_feat_list.append(pd.read_hdf('../cache/str_download.hdf', 'w'))
	all_feat_list.append(pd.read_hdf('../cache/uid_download_feat.hdf', 'w'))
	all_feat_list.append(pd.read_hdf('../cache/uid_entropy_feat.hdf', 'w'))
	all_feat_list.append(pd.read_hdf('../cache/uid_count_feat.hdf', 'w'))
	all_feat_list.append(pd.read_hdf('../cache/str_uid_aid_decomposition_feat.hdf', 'w'))
	all_feat_list.append(pd.read_hdf('../cache/str_aid_uid_decomposition.hdf', 'w'))
	all_feat_list.append(pd.read_hdf('../cache/str_len.hdf', 'w'))
	
	all_feat_list.append(pd.read_hdf('../cache/topn_kw2.hdf', 'w'))
	# all_feat_list.append(pd.read_hdf('../cache/topn_interest2.hdf', 'w'))
	all_feat_list.append(pd.read_hdf('../cache/topn_topic2.hdf', 'w'))

	all_feat_list.append(pd.read_hdf('../cache/str_d2v.hdf', 'w'))
	all_feat_list.append(pd.read_hdf('../cache/str_w2v.hdf', 'w'))

	all_feat_list.append(pd.read_hdf('../cache/str_uid_uid_aid_decomposition.hdf', 'w'))
	all_feat_list.append(pd.read_hdf('../cache/str_aid_uid_aid_decomposition.hdf', 'w'))

	all_feat_list.append(pd.read_hdf('../cache/str_decomposition_fuck.hdf', 'w'))
#
# all_feat_list.append(pd.read_hdf('../cache/str_aid_decomposition.hdf', 'w'))
# all_feat_list.append(pd.read_hdf('../cache/str_decomposition.hdf', 'w'))
# all_feat_list.append(pd.read_hdf('../cache1/str_label_feat_alltopic1.hdf', 'w'))
# all_feat_list.append(pd.read_hdf('../cache1/str_label_feat_alltopic2.hdf', 'w'))
# all_feat_list.append(pd.read_hdf('../cache1/str_label_feat_allkw1.hdf', 'w'))
# all_feat_list.append(pd.read_hdf('../cache1/str_label_feat_allkw2.hdf', 'w'))
# all_feat_list.append(pd.read_hdf('../cache1/str_label_feat_allinterest5.hdf', 'w'))
# all_feat_list.append(pd.read_hdf('../cache1/str_label_feat_allinterest2.hdf', 'w'))
# all_feat_list.append(pd.read_hdf('../cache1/str_label_feat_allinterest1.hdf', 'w'))


if a == '1':
	# str_decomposition = pd.read_hdf('../cache/str_decomposition.hdf','w')
	# str_feat = ['interest1', 'interest2', 'interest5', 'kw1', 'kw2', 'topic1', 'topic2']
	# for each in str_decomposition.columns:
	# 	for fuck in str_feat:
	# 		if fuck in each:
	# 			all_feat_list.append(str_decomposition[[each]])
	
	# all_feat_list.append(pd.read_hdf('../cache/feat5cache.hdf', 'w'))
	# all_feat_list.append(pd.read_hdf('../cache/xin_feat.hdf', 'w'))
	all_feat_list.append(pd.read_hdf('../cache/str_download.hdf','w'))
	all_feat_list.append(pd.read_hdf('../cache/uid_download_feat.hdf', 'w'))
	all_feat_list.append(pd.read_hdf('../cache/uid_entropy_feat.hdf','w'))
	all_feat_list.append(pd.read_hdf('../cache/uid_count_feat.hdf', 'w'))
	all_feat_list.append(pd.read_hdf('../cache/str_uid_aid_decomposition_feat.hdf','w'))
	all_feat_list.append(pd.read_hdf('../cache/str_aid_uid_decomposition.hdf','w'))
	all_feat_list.append(pd.read_hdf('../cache/str_len.hdf','w'))
	all_feat_list.append(pd.read_hdf('../cache/str_d2v.hdf', 'w'))
	all_feat_list.append(pd.read_hdf('../cache/str_w2v.hdf', 'w'))

	all_feat_list.append(pd.read_hdf('../cache/str_uid_uid_aid_decomposition.hdf', 'w'))
	all_feat_list.append(pd.read_hdf('../cache/str_aid_uid_aid_decomposition.hdf', 'w'))

	all_feat_list.append(pd.read_hdf('../cache/str_decomposition_fuck.hdf', 'w'))
	
	# all_feat_list.append(pd.read_hdf('../cache1/str_label_feat_alltopic1.hdf','w'))
	# all_feat_list.append(pd.read_hdf('../cache1/str_label_feat_alltopic2.hdf','w'))
	# all_feat_list.append(pd.read_hdf('../cache1/str_label_feat_allkw1.hdf','w'))
	# all_feat_list.append(pd.read_hdf('../cache1/str_label_feat_allkw2.hdf','w'))
	# all_feat_list.append(pd.read_hdf('../cache1/str_label_feat_allinterest5.hdf','w'))
	# all_feat_list.append(pd.read_hdf('../cache1/str_label_feat_allinterest2.hdf','w'))
	# all_feat_list.append(pd.read_hdf('../cache1/str_label_feat_allinterest1.hdf','w'))
	# all_feat_list.append(pd.read_hdf('../cache/str_download_means.hdf','w'))


if a == '2':
	str_decomposition = pd.read_hdf('../cache/str_decomposition.hdf','w')
	str_feat = ['interest1', 'interest2', 'interest5', 'kw1', 'kw2', 'topic1', 'topic2']
	for each in str_decomposition.columns:
		for fuck in str_feat:
			if fuck in each:
				all_feat_list.append(str_decomposition[[each]])
	
	# all_feat_list.append(pd.read_hdf('../cache/feat5cache.hdf', 'w'))
	# all_feat_list.append(pd.read_hdf('../cache/xin_feat.hdf', 'w'))
	all_feat_list.append(pd.read_hdf('../cache/str_download.hdf', 'w'))
	all_feat_list.append(pd.read_hdf('../cache/uid_download_feat.hdf', 'w'))
	all_feat_list.append(pd.read_hdf('../cache/uid_entropy_feat.hdf', 'w'))
	all_feat_list.append(pd.read_hdf('../cache/uid_count_feat.hdf', 'w'))
	all_feat_list.append(pd.read_hdf('../cache/str_uid_aid_decomposition_feat.hdf', 'w'))
	all_feat_list.append(pd.read_hdf('../cache/str_aid_uid_decomposition.hdf', 'w'))
	all_feat_list.append(pd.read_hdf('../cache/str_len.hdf', 'w'))
	
	all_feat_list.append(pd.read_hdf('../cache/topn_kw2.hdf', 'w'))
	all_feat_list.append(pd.read_hdf('../cache/topn_interest2.hdf', 'w'))
	all_feat_list.append(pd.read_hdf('../cache/topn_topic2.hdf', 'w'))
	
	all_feat_list.append(pd.read_hdf('../cache/str_d2v.hdf', 'w'))
	all_feat_list.append(pd.read_hdf('../cache/str_w2v.hdf', 'w'))

	all_feat_list.append(pd.read_hdf('../cache/str_uid_uid_aid_decomposition.hdf', 'w'))
	all_feat_list.append(pd.read_hdf('../cache/str_aid_uid_aid_decomposition.hdf', 'w'))

	all_feat_list.append(pd.read_hdf('../cache/str_decomposition_fuck.hdf', 'w'))

	# all_feat_list.append(pd.read_hdf('../cache1/str_label_feat_alltopic1.hdf','w'))
	# all_feat_list.append(pd.read_hdf('../cache1/str_label_feat_alltopic2.hdf','w'))
	# all_feat_list.append(pd.read_hdf('../cache1/str_label_feat_allkw1.hdf','w'))
	# all_feat_list.append(pd.read_hdf('../cache1/str_label_feat_allkw2.hdf','w'))
	# all_feat_list.append(pd.read_hdf('../cache1/str_label_feat_allinterest5.hdf','w'))
	# all_feat_list.append(pd.read_hdf('../cache1/str_label_feat_allinterest2.hdf','w'))
	# all_feat_list.append(pd.read_hdf('../cache1/str_label_feat_allinterest1.hdf','w'))
	# all_feat_list.append(pd.read_hdf('../cache/str_download_means.hdf','w'))

if a == '3':
	# str_decomposition = pd.read_hdf('../cache/str_decomposition.hdf', 'w')
	# str_feat = ['interest1', 'interest2', 'interest5', 'kw1', 'kw2', 'topic1', 'topic2']
	# for each in str_decomposition.columns:
	# 	for fuck in str_feat:
	# 		if fuck in each:
	# 			all_feat_list.append(str_decomposition[[each]])
	
	# all_feat_list.append(pd.read_hdf('../cache/feat5cache.hdf', 'w'))
	# all_feat_list.append(pd.read_hdf('../cache/xin_feat.hdf', 'w'))
	all_feat_list.append(pd.read_hdf('../cache/str_download.hdf', 'w'))
	all_feat_list.append(pd.read_hdf('../cache/uid_download_feat.hdf', 'w'))
	all_feat_list.append(pd.read_hdf('../cache/uid_entropy_feat.hdf', 'w'))
	all_feat_list.append(pd.read_hdf('../cache/uid_count_feat.hdf', 'w'))
	all_feat_list.append(pd.read_hdf('../cache/str_uid_aid_decomposition_feat.hdf', 'w'))
	all_feat_list.append(pd.read_hdf('../cache/str_aid_uid_decomposition.hdf', 'w'))
	all_feat_list.append(pd.read_hdf('../cache/str_len.hdf', 'w'))
	
	all_feat_list.append(pd.read_hdf('../cache/topn_kw2.hdf', 'w'))
	all_feat_list.append(pd.read_hdf('../cache/topn_interest2.hdf', 'w'))
	all_feat_list.append(pd.read_hdf('../cache/topn_topic2.hdf', 'w'))

	all_feat_list.append(pd.read_hdf('../cache1/str_label_feat_alltopic1.hdf','w'))
	all_feat_list.append(pd.read_hdf('../cache1/str_label_feat_alltopic2.hdf','w'))
	all_feat_list.append(pd.read_hdf('../cache1/str_label_feat_allkw1.hdf','w'))
	all_feat_list.append(pd.read_hdf('../cache1/str_label_feat_allkw2.hdf','w'))
	all_feat_list.append(pd.read_hdf('../cache1/str_label_feat_allinterest5.hdf','w'))
	all_feat_list.append(pd.read_hdf('../cache1/str_label_feat_allinterest2.hdf','w'))
	all_feat_list.append(pd.read_hdf('../cache1/str_label_feat_allinterest1.hdf','w'))
	all_feat_list.append(pd.read_hdf('../cache/str_d2v.hdf', 'w'))
	all_feat_list.append(pd.read_hdf('../cache/str_w2v.hdf', 'w'))

	all_feat_list.append(pd.read_hdf('../cache/str_uid_uid_aid_decomposition.hdf', 'w'))
	all_feat_list.append(pd.read_hdf('../cache/str_aid_uid_aid_decomposition.hdf', 'w'))

	all_feat_list.append(pd.read_hdf('../cache/str_decomposition_fuck.hdf', 'w'))

if a == '4':
	# str_decomposition = pd.read_hdf('../cache/str_decomposition.hdf', 'w')
	# str_feat = ['interest1', 'interest2', 'interest5', 'kw1', 'kw2', 'topic1', 'topic2']
	# for each in str_decomposition.columns:
	# 	for fuck in str_feat:
	# 		if fuck in each:
	# 			all_feat_list.append(str_decomposition[[each]])
	
	# all_feat_list.append(pd.read_hdf('../cache/feat5cache.hdf', 'w'))
	# all_feat_list.append(pd.read_hdf('../cache/xin_feat.hdf', 'w'))
	all_feat_list.append(pd.read_hdf('../cache/str_download.hdf', 'w'))
	all_feat_list.append(pd.read_hdf('../cache/uid_download_feat.hdf', 'w'))
	all_feat_list.append(pd.read_hdf('../cache/uid_entropy_feat.hdf', 'w'))
	all_feat_list.append(pd.read_hdf('../cache/uid_count_feat.hdf', 'w'))
	all_feat_list.append(pd.read_hdf('../cache/str_uid_aid_decomposition_feat.hdf', 'w'))
	all_feat_list.append(pd.read_hdf('../cache/str_aid_uid_decomposition.hdf', 'w'))
	all_feat_list.append(pd.read_hdf('../cache/str_len.hdf', 'w'))
	
	all_feat_list.append(pd.read_hdf('../cache/topn_kw2.hdf', 'w'))
	# all_feat_list.append(pd.read_hdf('../cache/topn_interest2.hdf', 'w'))
	# all_feat_list.append(pd.read_hdf('../cache/topn_topic2.hdf', 'w'))
	
	# all_feat_list.append(pd.read_hdf('../cache1/str_label_feat_alltopic1.hdf', 'w'))
	# all_feat_list.append(pd.read_hdf('../cache1/str_label_feat_alltopic2.hdf', 'w'))
	# all_feat_list.append(pd.read_hdf('../cache1/str_label_feat_allkw1.hdf', 'w'))
	# all_feat_list.append(pd.read_hdf('../cache1/str_label_feat_allkw2.hdf', 'w'))
	# all_feat_list.append(pd.read_hdf('../cache1/str_label_feat_allinterest5.hdf', 'w'))
	# all_feat_list.append(pd.read_hdf('../cache1/str_label_feat_allinterest2.hdf', 'w'))
	# all_feat_list.append(pd.read_hdf('../cache1/str_label_feat_allinterest1.hdf', 'w'))
	# all_feat_list.append(pd.read_hdf('../cache/str_download_means.hdf', 'w'))
	all_feat_list.append(pd.read_hdf('../cache/str_d2v.hdf', 'w'))
	all_feat_list.append(pd.read_hdf('../cache/str_w2v.hdf', 'w'))

	all_feat_list.append(pd.read_hdf('../cache/str_uid_uid_aid_decomposition.hdf', 'w'))
	all_feat_list.append(pd.read_hdf('../cache/str_aid_uid_aid_decomposition.hdf', 'w'))

	all_feat_list.append(pd.read_hdf('../cache/str_decomposition_fuck.hdf', 'w'))
if a == '5':
	# str_decomposition = pd.read_hdf('../cache/str_decomposition.hdf', 'w')
	# str_feat = ['interest1', 'interest2', 'interest5', 'kw1', 'kw2', 'topic1', 'topic2']
	# for each in str_decomposition.columns:
	# 	for fuck in str_feat:
	# 		if fuck in each:
	# 			all_feat_list.append(str_decomposition[[each]])
	
	# all_feat_list.append(pd.read_hdf('../cache/feat5cache.hdf', 'w'))
	# all_feat_list.append(pd.read_hdf('../cache/xin_feat.hdf', 'w'))
	all_feat_list.append(pd.read_hdf('../cache/str_download.hdf', 'w'))
	all_feat_list.append(pd.read_hdf('../cache/uid_download_feat.hdf', 'w'))
	all_feat_list.append(pd.read_hdf('../cache/uid_entropy_feat.hdf', 'w'))
	all_feat_list.append(pd.read_hdf('../cache/uid_count_feat.hdf', 'w'))
	all_feat_list.append(pd.read_hdf('../cache/str_uid_aid_decomposition_feat.hdf', 'w'))
	all_feat_list.append(pd.read_hdf('../cache/str_aid_uid_decomposition.hdf', 'w'))
	all_feat_list.append(pd.read_hdf('../cache/str_len.hdf', 'w'))
	
	# all_feat_list.append(pd.read_hdf('../cache/topn_kw2.hdf', 'w'))
	all_feat_list.append(pd.read_hdf('../cache/topn_interest2.hdf', 'w'))
	all_feat_list.append(pd.read_hdf('../cache/str_d2v.hdf', 'w'))
	all_feat_list.append(pd.read_hdf('../cache/str_w2v.hdf', 'w'))

	all_feat_list.append(pd.read_hdf('../cache/str_uid_uid_aid_decomposition.hdf', 'w'))
	all_feat_list.append(pd.read_hdf('../cache/str_aid_uid_aid_decomposition.hdf', 'w'))

	all_feat_list.append(pd.read_hdf('../cache/str_decomposition_fuck.hdf', 'w'))
	# all_feat_list.append(pd.read_hdf('../cache/topn_topic2.hdf', 'w'))
	
	# all_feat_list.append(pd.read_hdf('../cache1/str_label_feat_alltopic1.hdf', 'w'))
	# all_feat_list.append(pd.read_hdf('../cache1/str_label_feat_alltopic2.hdf', 'w'))
	# all_feat_list.append(pd.read_hdf('../cache1/str_label_feat_allkw1.hdf', 'w'))
	# all_feat_list.append(pd.read_hdf('../cache1/str_label_feat_allkw2.hdf', 'w'))
	# all_feat_list.append(pd.read_hdf('../cache1/str_label_feat_allinterest5.hdf', 'w'))
	# all_feat_list.append(pd.read_hdf('../cache1/str_label_feat_allinterest2.hdf', 'w'))
	# all_feat_list.append(pd.read_hdf('../cache1/str_label_feat_allinterest1.hdf', 'w'))
	# all_feat_list.append(pd.read_hdf('../cache/str_download_means.hdf', 'w'))

if a == '6':
	# str_decomposition = pd.read_hdf('../cache/str_decomposition.hdf', 'w')
	# str_feat = ['interest1', 'interest2', 'interest5', 'kw1', 'kw2', 'topic1', 'topic2']
	# for each in str_decomposition.columns:
	# 	for fuck in str_feat:
	# 		if fuck in each:
	# 			all_feat_list.append(str_decomposition[[each]])
	
	# all_feat_list.append(pd.read_hdf('../cache/feat5cache.hdf', 'w'))
	# all_feat_list.append(pd.read_hdf('../cache/xin_feat.hdf', 'w'))
	all_feat_list.append(pd.read_hdf('../cache/str_download.hdf', 'w'))
	all_feat_list.append(pd.read_hdf('../cache/uid_download_feat.hdf', 'w'))
	all_feat_list.append(pd.read_hdf('../cache/uid_entropy_feat.hdf', 'w'))
	all_feat_list.append(pd.read_hdf('../cache/uid_count_feat.hdf', 'w'))
	all_feat_list.append(pd.read_hdf('../cache/str_uid_aid_decomposition_feat.hdf', 'w'))
	all_feat_list.append(pd.read_hdf('../cache/str_aid_uid_decomposition.hdf', 'w'))
	all_feat_list.append(pd.read_hdf('../cache/str_len.hdf', 'w'))
	all_feat_list.append(pd.read_hdf('../cache/str_d2v.hdf', 'w'))
	all_feat_list.append(pd.read_hdf('../cache/str_w2v.hdf', 'w'))

	all_feat_list.append(pd.read_hdf('../cache/str_uid_uid_aid_decomposition.hdf', 'w'))
	all_feat_list.append(pd.read_hdf('../cache/str_aid_uid_aid_decomposition.hdf', 'w'))

	all_feat_list.append(pd.read_hdf('../cache/str_decomposition_fuck.hdf', 'w'))
	
	# all_feat_list.append(pd.read_hdf('../cache/topn_kw2.hdf', 'w'))
	all_feat_list.append(pd.read_hdf('../cache/topn_interest2.hdf', 'w'))
	# all_feat_list.append(pd.read_ hdf('../cache/topn_topic2.hdf', 'w'))

# all_feat_list.append(pd.read_hdf('../cache1/str_label_feat_alltopic1.hdf', 'w'))
# all_feat_list.append(pd.read_hdf('../cache1/str_label_feat_alltopic2.hdf', 'w'))
# all_feat_list.append(pd.read_hdf('../cache1/str_label_feat_allkw1.hdf', 'w'))
# all_feat_list.append(pd.read_hdf('../cache1/str_label_feat_allkw2.hdf', 'w'))
# all_feat_list.append(pd.read_hdf('../cache1/str_label_feat_allinterest5.hdf', 'w'))
# all_feat_list.append(pd.read_hdf('../cache1/str_label_feat_allinterest2.hdf', 'w'))
# all_feat_list.append(pd.read_hdf('../cache1/str_label_feat_allinterest1.hdf', 'w'))
# all_feat_list.append(pd.read_hdf('../cache/str_download_means.hdf', 'w'))


if a == '7':
	# str_decomposition = pd.read_hdf('../cache/str_decomposition.hdf', 'w')
	# str_feat = ['interest1', 'interest2', 'interest5', 'kw1', 'kw2', 'topic1', 'topic2']
	# for each in str_decomposition.columns:
	# 	for fuck in str_feat:
	# 		if fuck in each:
	# 			all_feat_list.append(str_decomposition[[each]])
	
	# all_feat_list.append(pd.read_hdf('../cache/feat5cache.hdf', 'w'))
	# all_feat_list.append(pd.read_hdf('../cache/xin_feat.hdf', 'w'))
	all_feat_list.append(pd.read_hdf('../cache/str_download.hdf', 'w'))
	all_feat_list.append(pd.read_hdf('../cache/uid_download_feat.hdf', 'w'))
	all_feat_list.append(pd.read_hdf('../cache/uid_entropy_feat.hdf', 'w'))
	all_feat_list.append(pd.read_hdf('../cache/uid_count_feat.hdf', 'w'))
	all_feat_list.append(pd.read_hdf('../cache/str_uid_aid_decomposition_feat.hdf', 'w'))
	all_feat_list.append(pd.read_hdf('../cache/str_aid_uid_decomposition.hdf', 'w'))
	all_feat_list.append(pd.read_hdf('../cache/str_len.hdf', 'w'))
	all_feat_list.append(pd.read_hdf('../cache/str_d2v.hdf', 'w'))
	all_feat_list.append(pd.read_hdf('../cache/str_w2v.hdf', 'w'))

	all_feat_list.append(pd.read_hdf('../cache/str_uid_uid_aid_decomposition.hdf', 'w'))
	all_feat_list.append(pd.read_hdf('../cache/str_aid_uid_aid_decomposition.hdf', 'w'))

	all_feat_list.append(pd.read_hdf('../cache/str_decomposition_fuck.hdf', 'w'))
	# all_feat_list.append(pd.read_hdf('../cache/topn_kw2.hdf', 'w'))
	# all_feat_list.append(pd.read_hdf('../cache/topn_interest2.hdf', 'w'))
	all_feat_list.append(pd.read_hdf('../cache/topn_topic2.hdf', 'w'))

# all_feat_list.append(pd.read_hdf('../cache1/str_label_feat_alltopic1.hdf', 'w'))
# all_feat_list.append(pd.read_hdf('../cache1/str_label_feat_alltopic2.hdf', 'w'))
# all_feat_list.append(pd.read_hdf('../cache1/str_label_feat_allkw1.hdf', 'w'))
# all_feat_list.append(pd.read_hdf('../cache1/str_label_feat_allkw2.hdf', 'w'))
# all_feat_list.append(pd.read_hdf('../cache1/str_label_feat_allinterest5.hdf', 'w'))
# all_feat_list.append(pd.read_hdf('../cache1/str_label_feat_allinterest2.hdf', 'w'))
# all_feat_list.append(pd.read_hdf('../cache1/str_label_feat_allinterest1.hdf', 'w'))
# all_feat_list.append(pd.read_hdf('../cache/str_download_means.hdf', 'w'))


if a == '8':
	# str_decomposition = pd.read_hdf('../cache/str_decomposition.hdf', 'w')
	# str_feat = ['interest1', 'interest2', 'interest5', 'kw1', 'kw2', 'topic1', 'topic2']
	# for each in str_decomposition.columns:
	# 	for fuck in str_feat:
	# 		if fuck in each:
	# 			all_feat_list.append(str_decomposition[[each]])
	
	# all_feat_list.append(pd.read_hdf('../cache/feat5cache.hdf', 'w'))
	# all_feat_list.append(pd.read_hdf('../cache/xin_feat.hdf', 'w'))
	all_feat_list.append(pd.read_hdf('../cache/str_download.hdf', 'w'))
	all_feat_list.append(pd.read_hdf('../cache/uid_download_feat.hdf', 'w'))
	all_feat_list.append(pd.read_hdf('../cache/uid_entropy_feat.hdf', 'w'))
	all_feat_list.append(pd.read_hdf('../cache/uid_count_feat.hdf', 'w'))
	all_feat_list.append(pd.read_hdf('../cache/str_uid_aid_decomposition_feat.hdf', 'w'))
	all_feat_list.append(pd.read_hdf('../cache/str_aid_uid_decomposition.hdf', 'w'))
	all_feat_list.append(pd.read_hdf('../cache/str_len.hdf', 'w'))
	all_feat_list.append(pd.read_hdf('../cache/str_d2v.hdf', 'w'))
	all_feat_list.append(pd.read_hdf('../cache/str_w2v.hdf', 'w'))

	all_feat_list.append(pd.read_hdf('../cache/str_uid_uid_aid_decomposition.hdf', 'w'))
	all_feat_list.append(pd.read_hdf('../cache/str_aid_uid_aid_decomposition.hdf', 'w'))

	all_feat_list.append(pd.read_hdf('../cache/str_decomposition_fuck.hdf', 'w'))
	# all_feat_list.append(pd.read_hdf('../cache/topn_kw2.hdf', 'w'))
	# all_feat_list.append(pd.read_hdf('../cache/topn_interest2.hdf', 'w'))
	all_feat_list.append(pd.read_hdf('../cache/topn_topic2.hdf', 'w'))
	
	all_feat_list.append(pd.read_hdf('../cache/str_aid_decomposition.hdf', 'w'))
	all_feat_list.append(pd.read_hdf('../cache/str_decomposition.hdf', 'w'))
# all_feat_list.append(pd.read_hdf('../cache1/str_label_feat_alltopic1.hdf', 'w'))
# all_feat_list.append(pd.read_hdf('../cache1/str_label_feat_alltopic2.hdf', 'w'))
# all_feat_list.append(pd.read_hdf('../cache1/str_label_feat_allkw1.hdf', 'w'))
# all_feat_list.append(pd.read_hdf('../cache1/str_label_feat_allkw2.hdf', 'w'))
# all_feat_list.append(pd.read_hdf('../cache1/str_label_feat_allinterest5.hdf', 'w'))
# all_feat_list.append(pd.read_hdf('../cache1/str_label_feat_allinterest2.hdf', 'w'))
# all_feat_list.append(pd.read_hdf('../cache1/str_label_feat_allinterest1.hdf', 'w'))
# all_feat_list.append(pd.read_hdf('../cache/str_download_means.hdf', 'w'))


if a == '9':
	# str_decomposition = pd.read_hdf('../cache/str_decomposition.hdf', 'w')
	# str_feat = ['interest1', 'interest2', 'interest5', 'kw1', 'kw2', 'topic1', 'topic2']
	# for each in str_decomposition.columns:
	# 	for fuck in str_feat:
	# 		if fuck in each:
	# 			all_feat_list.append(str_decomposition[[each]])
	
	# all_feat_list.append(pd.read_hdf('../cache/feat5cache.hdf', 'w'))
	# all_feat_list.append(pd.read_hdf('../cache/xin_feat.hdf', 'w'))
	all_feat_list.append(pd.read_hdf('../cache/str_download.hdf', 'w'))
	all_feat_list.append(pd.read_hdf('../cache/uid_download_feat.hdf', 'w'))
	all_feat_list.append(pd.read_hdf('../cache/uid_entropy_feat.hdf', 'w'))
	all_feat_list.append(pd.read_hdf('../cache/uid_count_feat.hdf', 'w'))
	all_feat_list.append(pd.read_hdf('../cache/str_uid_aid_decomposition_feat.hdf', 'w'))
	all_feat_list.append(pd.read_hdf('../cache/str_aid_uid_decomposition.hdf', 'w'))
	all_feat_list.append(pd.read_hdf('../cache/str_len.hdf', 'w'))
	
	# all_feat_list.append(pd.read_hdf('../cache/topn_kw2.hdf', 'w'))
	# all_feat_list.append(pd.read_hdf('../cache/topn_interest2.hdf', 'w'))
	all_feat_list.append(pd.read_hdf('../cache/topn_topic2.hdf', 'w'))
	all_feat_list.append(pd.read_hdf('../cache/str_d2v.hdf', 'w'))
	all_feat_list.append(pd.read_hdf('../cache/str_w2v.hdf', 'w'))

	all_feat_list.append(pd.read_hdf('../cache/str_uid_uid_aid_decomposition.hdf', 'w'))
	all_feat_list.append(pd.read_hdf('../cache/str_aid_uid_aid_decomposition.hdf', 'w'))

	all_feat_list.append(pd.read_hdf('../cache/str_decomposition_fuck.hdf', 'w'))
	all_feat_list.append(pd.read_hdf('../cache/str_aid_decomposition.hdf', 'w'))
	# all_feat_list.append(pd.read_hdf('../cache/str_decomposition.hdf', 'w'))
# all_feat_list.append(pd.read_hdf('../cache1/str_label_feat_alltopic1.hdf', 'w'))
# all_feat_list.append(pd.read_hdf('../cache1/str_label_feat_alltopic2.hdf', 'w'))
# all_feat_list.append(pd.read_hdf('../cache1/str_label_feat_allkw1.hdf', 'w'))
# all_feat_list.append(pd.read_hdf('../cache1/str_label_feat_allkw2.hdf', 'w'))
# all_feat_list.append(pd.read_hdf('../cache1/str_label_feat_allinterest5.hdf', 'w'))
# all_feat_list.append(pd.read_hdf('../cache1/str_label_feat_allinterest2.hdf', 'w'))
# all_feat_list.append(pd.read_hdf('../cache1/str_label_feat_allinterest1.hdf', 'w'))
# all_feat_list.append(pd.read_hdf('../cache/str_download_means.hdf', 'w'))



if a == '10':
	# str_decomposition = pd.read_hdf('../cache/str_decomposition.hdf', 'w')
	# str_feat = ['interest1', 'interest2', 'interest5', 'kw1', 'kw2', 'topic1', 'topic2']
	# for each in str_decomposition.columns:
	# 	for fuck in str_feat:
	# 		if fuck in each:
	# 			all_feat_list.append(str_decomposition[[each]])
	
	# all_feat_list.append(pd.read_hdf('../cache/feat5cache.hdf', 'w'))
	# all_feat_list.append(pd.read_hdf('../cache/xin_feat.hdf', 'w'))
	all_feat_list.append(pd.read_hdf('../cache/str_download.hdf', 'w'))
	all_feat_list.append(pd.read_hdf('../cache/uid_download_feat.hdf', 'w'))
	all_feat_list.append(pd.read_hdf('../cache/uid_entropy_feat.hdf', 'w'))
	all_feat_list.append(pd.read_hdf('../cache/uid_count_feat.hdf', 'w'))
	all_feat_list.append(pd.read_hdf('../cache/str_uid_aid_decomposition_feat.hdf', 'w'))
	all_feat_list.append(pd.read_hdf('../cache/str_aid_uid_decomposition.hdf', 'w'))
	all_feat_list.append(pd.read_hdf('../cache/str_len.hdf', 'w'))
	
	all_feat_list.append(pd.read_hdf('../cache/topn_kw2.hdf', 'w'))
	all_feat_list.append(pd.read_hdf('../cache/topn_interest2.hdf', 'w'))
	all_feat_list.append(pd.read_hdf('../cache/topn_topic2.hdf', 'w'))
	
	all_feat_list.append(pd.read_hdf('../cache/str_aid_decomposition.hdf', 'w'))
	all_feat_list.append(pd.read_hdf('../cache/str_decomposition.hdf', 'w'))
	all_feat_list.append(pd.read_hdf('../cache1/str_label_feat_alltopic1.hdf', 'w'))
	all_feat_list.append(pd.read_hdf('../cache1/str_label_feat_alltopic2.hdf', 'w'))
	all_feat_list.append(pd.read_hdf('../cache1/str_label_feat_allkw1.hdf', 'w'))
	all_feat_list.append(pd.read_hdf('../cache1/str_label_feat_allkw2.hdf', 'w'))
	all_feat_list.append(pd.read_hdf('../cache1/str_label_feat_allinterest5.hdf', 'w'))
	all_feat_list.append(pd.read_hdf('../cache1/str_label_feat_allinterest2.hdf', 'w'))
	all_feat_list.append(pd.read_hdf('../cache1/str_label_feat_allinterest1.hdf', 'w'))
	all_feat_list.append(pd.read_hdf('../cache/str_d2v.hdf', 'w'))
	all_feat_list.append(pd.read_hdf('../cache/str_w2v.hdf', 'w'))

	all_feat_list.append(pd.read_hdf('../cache/str_uid_uid_aid_decomposition.hdf', 'w'))
	all_feat_list.append(pd.read_hdf('../cache/str_aid_uid_aid_decomposition.hdf', 'w'))

	all_feat_list.append(pd.read_hdf('../cache/str_decomposition_fuck.hdf', 'w'))
# all_feat_list.append(pd.read_hdf('../cache/str_download_means.hdf', 'w'))


	
	
for each in all_feat_list:
	print(each.shape)
all_feat = concat(all_feat_list)
del all_feat_list
gc.collect()
print('concat down!')
# str_feat = ['interest1', 'interest2', 'interest3', 'interest4', 'interest5', 'kw1', 'kw2', 'kw3', 'topic1',
# 		            'topic2', 'topic3', 'appIdInstall', 'appIdAction']


# get_second_add(all_feat,['entropy'])
# get_second_add(all_feat,['_download_rate'])
# get_second_add(all_feat,['_download_rate','LBS'])
# get_second_add(all_feat,['_download_rate','age'])
# get_second_add(all_feat,['_download_rate','consumptionAbility'])
# get_second_add(all_feat,['_download_rate','gender'])
# get_second_add(all_feat,['_download_rate','aid'])
# get_second_add(all_feat,['_download_rate','_productId'])
# get_second_add(all_feat,['_download_rate','marriageStatus'])
#
# get_second_add(all_feat,['_aid_max_download','interest'])
# get_second_add(all_feat,['_aid_max_download','kw'])
# get_second_add(all_feat,['_aid_max_download','topic'])
# get_second_add(all_feat,['_aid_num_download','interest'])
# get_second_add(all_feat,['_aid_num_download','kw'])
# get_second_add(all_feat,['_aid_num_download','topic'])
# get_second_add(all_feat,['_aid_mean_download','interest'])
# get_second_add(all_feat,['_aid_mean_download','kw'])
# get_second_add(all_feat,['_aid_mean_download','topic'])
# # get_second_add(all_feat,['_aid_sum_mean_download','interest'])
# # get_second_add(all_feat,['_aid_sum_mean_download','kw'])
# # get_second_add(all_feat,['_aid_sum_mean_download','topic'])
#
# get_second_add(all_feat,['_aid_max_download','interest','aid'])
# get_second_add(all_feat,['_aid_max_download','kw','aid'])
# get_second_add(all_feat,['_aid_max_download','topic','aid'])
# get_second_add(all_feat,['_aid_num_download','interest','aid'])
# get_second_add(all_feat,['_aid_num_download','kw','aid'])
# get_second_add(all_feat,['_aid_num_download','topic','aid'])
# get_second_add(all_feat,['_aid_mean_download','interest','aid'])
# get_second_add(all_feat,['_aid_mean_download','kw','aid'])
# get_second_add(all_feat,['_aid_mean_download','topic','aid'])
# # get_second_add(all_feat,['_aid_sum_mean_download','interest','aid'])
# # get_second_add(all_feat,['_aid_sum_mean_download','kw','aid'])
# # get_second_add(all_feat,['_aid_sum_mean_download','topic','aid'])
#
# get_second_add(all_feat,['len_download_rate'])
# get_second_add(all_feat,['len_aid_download_rate'])
# get_second_add(all_feat,['_aid_sum_mean_download'])
# get_second_add(all_feat,['_aid_num_download'])
# get_second_add(all_feat,['_aid_mean_download'])
# get_second_add(all_feat,['_aid_max_download'])
#
# # get_second_add(all_feat,['_label_feat_all','kw'])
# # get_second_add(all_feat,['_label_feat_all','topic'])
# # get_second_add(all_feat,['_label_feat_all','interest'])
# # get_second_add(all_feat,['_aid_label_feat_all','interest'])
# # get_second_add(all_feat,['_aid_label_feat_all','kw'])
# # get_second_add(all_feat,['_aid_label_feat_all','topic'])




train_feat = all_feat[all_feat['label'] != -1].sample(frac=1, random_state=233333333).reset_index(drop=True)

test_feat = all_feat[all_feat['label'] == -1]

del all_feat
gc.collect()
train_feat_eval_fold = train_feat[:int(train_feat.shape[0] * 0.1)]
train_dft_train_fold = train_feat[int(train_feat.shape[0] * 0.1):]
del train_feat
gc.collect()


# train_dft_train_fold_neg = train_dft_train_fold[train_dft_train_fold['label'] == 0].sample(frac=0.4,
#                                                                                            random_state=233).reset_index(
# 	drop=True)
#
# train_dft_train_fold_pos = train_dft_train_fold[train_dft_train_fold['label'] != 0]
#
# train_dft_train_fold = train_dft_train_fold_neg.append(train_dft_train_fold_pos).sample(frac=1,
#                                                                                         random_state=233).reset_index(
# 	drop=True)
#
# del train_dft_train_fold_pos
# del train_dft_train_fold_neg
# gc.collect()


target = 'label'

train_y = train_dft_train_fold[target]
del train_dft_train_fold[target]
eval_y = train_feat_eval_fold[target]
del train_feat_eval_fold[target]

predictors = [x for x in train_feat_eval_fold.columns]


# uid_num_feat = ['LBS', 'age', 'carrier', 'consumptionAbility', 'education', 'gender', 'house', 'os', 'ct',
#                 'marriageStatus']
#
# aid_feat = ['advertiserId', 'campaignId', 'creativeId', 'creativeSize', 'adCategoryId', 'productId', 'productType']



cat_col = [
	# 'aid_re','advertiserId', 'campaignId', 'creativeId', 'creativeSize', 'adCategoryId', 'productId', 'productType',
     #       'LBS',
	# 'age', 'carrier', 'consumptionAbility', 'education', 'gender', 'house', 'os', 'ct',
           # 'marriageStatus'
]

categorical_feature_list = []
for i,each in enumerate(predictors):
	if each in cat_col:
		categorical_feature_list.append(i)
		print(predictors[i])

print("LGB make_Dataset")
lgb_train = lgb.Dataset(train_dft_train_fold, train_y,params = params,
                        categorical_feature=categorical_feature_list,
                        )
lgb_eval = lgb.Dataset(train_feat_eval_fold, eval_y,params = params,
                       categorical_feature=categorical_feature_list,
                       )

del train_dft_train_fold, train_y, train_feat_eval_fold, eval_y
gc.collect()
evals_results = {}
lgb_model = lgb.train(params,
                      lgb_train,
                      valid_sets=lgb_eval,
                      evals_result=evals_results,
                      num_boost_round=80000,
                      early_stopping_rounds=200,
                      verbose_eval=50,
					  categorical_feature=categorical_feature_list,
                      )
del lgb_train
del lgb_eval
gc.collect()
print('ok')
# sub
res = test_feat[['aid', 'uid']]
res['score'] = lgb_model.predict(test_feat[predictors])
res['score'] = res['score'].apply(lambda x: float('%.6f' % x))

test1 = pd.read_csv('../input/fu/test2.csv')
test1 = test1.merge(res,'left',['aid','uid'])
test1.to_csv('submission'+ a + the_data + '.csv',index=False)
# os.system('zip baseline_a.zip submission.csv')
#
#
#
# # imp
# feat_imp = pd.Series(lgb_model.feature_importance(), index=predictors).sort_values(ascending=False)
# feat_imp = pd.DataFrame(feat_imp)
# feat_imp.to_csv('../output/split_xin5.csv')
# feat_imp = pd.Series(lgb_model.feature_importance('gain'), index=predictors).sort_values(ascending=False)
# feat_imp = pd.DataFrame(feat_imp)
# feat_imp.to_csv('../output/gain_xin5.csv')
# gc.collect()
