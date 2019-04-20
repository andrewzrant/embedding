# coding=utf-8
import pandas as pd
import numpy as  np
import random
import zipfile
import os
import gc
from multiprocessing import cpu_count
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from scipy import sparse
import warnings
import scipy.special as special
from gensim.models import Word2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.decomposition import LatentDirichletAllocation, NMF, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
from scipy.stats import entropy
from util import *
import sys

warnings.filterwarnings("ignore")

print('feat5')
cache_path = '../cache/'
flag = 0
sub = True
the_data = sys.argv[2]

uid_num_feat = ['LBS', 'age', 'carrier', 'consumptionAbility', 'education', 'gender', 'house', 'os', 'ct',
                'marriageStatus']

aid_feat = ['advertiserId', 'campaignId', 'creativeId', 'creativeSize', 'adCategoryId', 'productId', 'productType']

cross_feat_list = [[['LBS', 'age', 'carrier', 'consumptionAbility', 'gender', 'os'], ['aid']],
                   [['LBS', 'marriageStatus', 'age', 'ct', 'house'], ['aid']],
                   [['age', 'carrier', 'consumptionAbility', 'gender', 'os', 'marriageStatus', 'education'], ['aid']],
                   [['aid'], ['LBS', 'consumptionAbility', 'house']],
                   [['aid'], ['age', 'carrier', 'consumptionAbility', 'gender', 'os', 'marriageStatus', 'education']],
                   [['aid'], ['age', 'carrier', 'gender', 'house', 'os', 'LBS']],
                   # [['uid'], ['aid']],
                   # [['aid'], ['uid']],
                   ]

count_rate_list = [[['gender', 'aid', 'consumptionAbility', 'carrier'], ['age', 'gender', 'aid']],
                   [['LBS', 'aid'], ['aid']],
                   [['LBS', 'age', 'carrier', 'consumptionAbility', 'education', 'gender', 'house', 'os',
                     'marriageStatus', 'aid'], ['aid']],
                   [['age', 'gender', 'house', 'aid'], ['aid', 'age']],
                   [['LBS', 'gender', 'age', 'aid'], ['aid', 'LBS']],
                   [['LBS', 'consumptionAbility', 'carrier'], ['aid', 'LBS']],
                   [['LBS', 'gender', 'education', 'age', 'aid'], ['LBS', 'age', 'aid']],
                   [['LBS', 'gender', 'education', 'aid'], ['LBS', 'aid']],
                   [['age', 'carrier', 'consumptionAbility', 'education', 'gender', 'marriageStatus', 'aid'], ['aid']],
                   [['LBS', 'age', 'gender', 'house', 'aid'], ['aid', 'age', 'LBS', ]],
                   [['gender', 'marriageStatus', 'age', 'aid'], ['aid', 'age']],
                   [['age', 'carrier', 'consumptionAbility', 'os', 'ct'], ['aid', 'age']],
                   ]

download_feat_list = ['aid']
for each in uid_num_feat:
	download_feat_list.append(each + '_aid')
	download_feat_list.append(each + '_productId')
	download_feat_list.append(each)

for each in aid_feat:
	# download_feat_list.append(each + '_uid')
	# count_rate_list.append([['uid'], ['uid', each]])
	# cross_feat_list.append([[each], ['uid']])
	# cross_feat_list.append([['uid'], [each]])
	download_feat_list.append(each + '_house_os_consumptionAbility_gender_marriageStatus_education_LBS_age')
	count_rate_list.append(
		[['house', 'os', 'consumptionAbility', 'gender', 'marriageStatus', 'education', 'LBS', 'age'],
		 ['house', 'os', 'consumptionAbility', 'gender', 'marriageStatus', 'education', 'LBS', 'age', each]])
	cross_feat_list.append(
		[[each], ['house', 'os', 'consumptionAbility', 'gender', 'marriageStatus', 'education', 'LBS', 'age']])
	cross_feat_list.append(
		[['house', 'os', 'consumptionAbility', 'gender', 'marriageStatus', 'education', 'LBS', 'age'], [each]])

download_cross_feat = ['LBS_age', 'age_carrier', 'carrier_consumptionAbility', 'consumptionAbility_LBS',
                       'education_gender',
                       'gender_os', 'os_marriageStatus', 'LBS_education', 'age_gender', 'carrier_os',
                       'consumptionAbility_marriageStatus']
download_3_cross_feat = ['LBS_house_gender', 'age_os_marriageStatus', 'carrier_consumptionAbility_education',
                         'LBS_age_carrier',
                         'house_os_consumptionAbility', 'gender_marriageStatus_education', 'LBS_os_education',
                         'house_marriageStatus_carrier', 'gender_age_consumptionAbility', 'gender_os_carrier',
                         'house_age_education', 'LBS_os_carrier', 'ct_gender_age']
for each in download_cross_feat:
	download_feat_list.append(each + '_aid')

for each in download_3_cross_feat:
	download_feat_list.append(each + '_aid')


def make_feat(a):
	if not os.path.exists('../cache/feat5cache.hdf') or 1:
		dtype_list = {
			'LBS': 'float16',
			'age': 'int8',
			'carrier': 'int8',
			'consumptionAbility': 'int8',
			'uid': 'int32',
			'education': 'int8',
			# 'gender': 'int8',
			'house': 'float16',
			'aid': 'int32',
			'advertiserId': 'int32',
			'campaignId': 'int32',
			'creativeId': 'int32',
			'creativeSize': 'int32',
			'adCategoryId': 'int32',
			'productId': 'int32',
			'productType': 'int32',
			
		}
		
		test1_pd = pd.read_csv('../input/'+the_data+'/test1.csv')
		test2_pd = pd.read_csv('../input/'+the_data+'/test2.csv')
		test_pd = test1_pd.append(test2_pd).reset_index()
		train_pd = pd.read_csv('../input/'+the_data+'/train.csv')
		test_pd['label'] = -1
		train_pd['label'] = train_pd['label'].map(lambda x: 0 if x == -1 else 1)
		train_pd = train_pd.append(test_pd)
		train_pd['label'] = train_pd['label'].astype('int8')
		
		# aid
		adFeature_pd = pd.read_csv('../input/'+the_data+'/adFeature.csv', dtype=dtype_list)
		one_hot_aid_feature = ['advertiserId', 'campaignId', 'creativeId', 'adCategoryId', 'productId', 'productType',
		                       'creativeSize']
		for feature in one_hot_aid_feature:
			try:
				adFeature_pd[feature] = LabelEncoder().fit_transform(adFeature_pd[feature].apply(int))
			except:
				adFeature_pd[feature] = LabelEncoder().fit_transform(adFeature_pd[feature])
		print(adFeature_pd.max())
		adFeature_pd[one_hot_aid_feature] = adFeature_pd[one_hot_aid_feature].astype('int16')
		adFeature_pd['aid'] = adFeature_pd['aid'].astype('int16')
		
		# uid
		num_feat = ['LBS', 'age', 'carrier', 'consumptionAbility', 'education', 'gender', 'house', 'os', 'ct',
		            'marriageStatus']
		userFeature_pd = pd.read_csv('../input/'+the_data+'/userFeature.csv', dtype=dtype_list,
		                             usecols=num_feat + ['uid'])
		userFeature_pd['gender'] = userFeature_pd['gender'].fillna(1).astype('int8')
		print(userFeature_pd.max())
		userFeature_pd['ct'] = userFeature_pd['ct'].fillna('-1').map(sort2str)
		userFeature_pd['marriageStatus'] = userFeature_pd['marriageStatus'].fillna('-1').map(sort2str)
		userFeature_pd['os'] = userFeature_pd['os'].fillna('-1').map(sort2str)
		userFeature_pd['LBS'] = userFeature_pd['LBS'].fillna(-1).astype('int16')
		userFeature_pd['house'] = userFeature_pd['house'].fillna(-1).astype('int8')
		# ct_list = []
		# for each in set(list(userFeature_pd['ct'])):
		#     ct_list += each.split(' ')
		# for each in set(ct_list):
		#     userFeature_pd['ct_OH_'+each] = userFeature_pd['ct'].map(lambda x:1 if each in x.split(' ') else 0).astype('int8')
		#
		# marriageStatus_list = []
		# for each in set(list(userFeature_pd['marriageStatus'])):
		#     marriageStatus_list += each.split(' ')
		# for each in set(marriageStatus_list):
		#     userFeature_pd['marriageStatus_OH_'+each] = userFeature_pd['marriageStatus'].map(lambda x:1 if each in x.split(' ') else 0).astype('int8')
		#
		# os_list = []
		# for each in set(list(userFeature_pd['os'])):
		#     os_list += each.split(' ')
		# for each in set(os_list):
		#     userFeature_pd['os_OH_'+each] = userFeature_pd['os'].map(lambda x:1 if each in x.split(' ') else 0).astype('int8')
		#
		for feature in ['ct', 'marriageStatus', 'os']:
			try:
				userFeature_pd[feature] = LabelEncoder().fit_transform(userFeature_pd[feature].fillna('-1').apply(str))
			except:
				userFeature_pd[feature] = LabelEncoder().fit_transform(userFeature_pd[feature].fillna('-1'))
			print(userFeature_pd[feature].max())
			userFeature_pd[feature] = userFeature_pd[feature].astype('int8')
		
		# merge
		train_pd = train_pd.merge(adFeature_pd, 'left', ['aid'])
		train_pd = train_pd.merge(userFeature_pd, 'left', ['uid'])
		del userFeature_pd
		del adFeature_pd
		train_pd.to_hdf('../cache/feat5cache.hdf', 'w')
	else:
		train_pd = pd.read_hdf('../cache/feat5cache.hdf', 'w')
	print(train_pd.shape)
	
	if a == '1' or sub:
		for each in cross_feat_list:
			name = '_'.join(each[0]) + '_by_' + '_'.join(each[1])
			print('entropy_feat:', name)
			_, _, _, \
			train_pd[name + 'entropy'] = entropy_feat(train_pd, each[0], each[1])
			
			# train_pd[name + 'max_rate'] = train_pd[name + 'max_rate'].astype('float16')
			# train_pd[name + 'max_value'] = train_pd[name + 'max_value'].astype('int16')
			# train_pd[name + 'nunique'] = train_pd[name + 'nunique'].astype('int32')
			train_pd[name + 'entropy'] = train_pd[name + 'entropy'].astype('float16')
		print(train_pd.shape)
		train_pd.to_hdf('../cache/uid_entropy_feat.hdf', 'w')
	
	if a == '1' or sub:
		for each in count_rate_list:
			name = '_'.join(each[0]) + '_by_' + '_'.join(each[1])
			print('get_count_rate:', name)
			train_pd[name + '_count_rate'] = get_count_rate(train_pd, train_pd, each[0], each[1])
			train_pd[name + '_count_rate'] = train_pd[name + '_count_rate'].astype('float16')
		print(train_pd.shape)
		the_name = [x for x in train_pd.columns if '_count_rate' in x]
		train_pd[the_name].to_hdf('../cache/uid_count_feat.hdf', 'w')
	
	if a == '2' or sub:
		for each in download_feat_list:
			print('download_feat:', each)
			_, train_pd[each + '_download_rate'] = each_down_load_feat(train_pd, each, 5, download_feat_xuan)
			train_pd[each + '_download_rate'] = train_pd[each + '_download_rate'].astype('float16')
			cols_list = each.split('_')
			if len(each.split('_')) > 3 and len(each.split('_')) < 5:
				temp = train_pd.groupby(['uid'], as_index=False)[each + '_download_rate'].agg(
					{each + '_download_rate_mean': 'mean'})
				train_pd = train_pd.merge(temp, 'left', ['uid']).fillna(0)
				temp = train_pd.groupby(['aid'], as_index=False)[each + '_download_rate_mean'].agg(
					{each + '_download_rate_mean_mean': 'mean'})
				train_pd = train_pd.merge(temp, 'left', ['aid']).fillna(0)
				train_pd[each + '_download_rate_mean'] = train_pd[each + '_download_rate_mean'].astype('float16')
				train_pd[each + '_download_rate_mean_mean'] = train_pd[each + '_download_rate_mean_mean'].astype(
					'float16')
		the_name = [x for x in train_pd.columns if 'download' in x]
		train_pd[the_name].to_hdf('../cache/uid_download_feat.hdf', 'w')
	
	if a == '3' or sub:
		aid_list = list(set(train_pd['aid'].values.tolist()))
		aid_dict = {}
		for i, each in enumerate(aid_list):
			aid_dict[each] = i + 1
		train_pd['aid_re'] = train_pd['aid'].map(aid_dict)
		temp = train_pd.groupby(['uid'], as_index=False)['aid'].agg({'xin_aid_uid_count': 'count'})
		train_pd = train_pd.merge(temp, 'left', ['uid']).fillna(0)
		dicts = {}
		for name, each in train_pd.groupby(['uid']):
			dicts[name] = list(set(each['aid_re'].values.tolist())) + ['-1'] * 5
		print('go')
		train_pd['another_aid_re_0'] = list(
			map(lambda x, y: dicts[y][0] if dicts[y][0] != x else dicts[y][1], train_pd['aid_re'], train_pd['uid']))
		train_pd['another_aid_re_1'] = list(
			map(lambda x, y: dicts[y][1] if dicts[y][0] != x or dicts[y][1] != x else dicts[y][1], train_pd['aid_re'],
			    train_pd['uid']))
		
		train_pd['xin_aid_uid_count'] = train_pd['xin_aid_uid_count'].astype('int8')
		train_pd['another_aid_re_0'] = train_pd['another_aid_re_0'].astype('int32')
		train_pd['another_aid_re_1'] = train_pd['another_aid_re_1'].astype('int32')
		print(train_pd[['aid_re', 'xin_aid_uid_count', 'another_aid_re_0', 'another_aid_re_1']])
		train_pd[['aid_re', 'xin_aid_uid_count', 'another_aid_re_0', 'another_aid_re_1']].to_hdf('../code/xin_feat.hdf',
		                                                                                         'w')
	
	# train_pd[each + '_download_num'] = train_pd[each + '_download_num'].astype('float16')
	# print(train_pd[each+'_download_rate'])
	#     exit()
	
	# num_feat = ['age', 'carrier', 'consumptionAbility', 'education', 'gender', 'house']
	# train_pd = pd.get_dummies(train_pd,prefix_sep='_OH_',columns=num_feat)
	# for each in train_pd.columns:
	#     if '_OH_' in each:
	#         train_pd[each] = train_pd[each].astype('int8')
	# gc.collect()
	# for each in num_feat+['os','marriageStatus','ct']:
	#     for each_col in train_pd.columns:
	#         if each+'_OH_' in each_col:
	#             temp = train_pd.groupby(['aid'],as_index=False)[each_col].agg({each_col+'_aid_mean':'mean'})
	#             train_pd = train_pd.merge(temp,'left',['aid']).fillna(0)
	#             temp = train_pd.groupby(['uid'],as_index=False)[each_col+'_aid_mean'].agg({each_col+'_aid_uid_mean':'mean'})
	#             train_pd = train_pd.merge(temp, 'left', ['uid']).fillna(0)
	#             train_pd[each_col + '_aid_mean'] = train_pd[each_col + '_aid_mean'].astype('float16')
	#             train_pd[each_col+'_aid_uid_mean'] = train_pd[each_col+'_aid_uid_mean'].astype('float16')
	#             print(each_col)
	#
	# for each in ['uid','LBS']:
	#     temp = decomposition_cross_feat(train_pd, 'aid', [each], 0, 0, 5)
	#     train_pd = concat([train_pd, temp])
	#     for i in range(5):
	#         temp = train_pd.groupby(['uid'], as_index=False)['svd_num_cross_aid_by_' + each + str(i)].agg(
	#             {'svd_num_cross_aid_by_' + each + str(i) + '_uid_mean': 'mean'})
	#         train_pd = train_pd.merge(temp, 'left', ['uid']).fillna(0)
	#         train_pd['svd_num_cross_aid_by_' + each + str(i) + '_uid_mean'] = train_pd[
	#             'svd_num_cross_aid_by_' + each + str(i) + '_uid_mean'].astype('float16')
	
	print(train_pd.info())
	return train_pd


a = sys.argv[1]
num_feat = make_feat(a)
print(num_feat.shape)
