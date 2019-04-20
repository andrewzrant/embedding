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
print('feat6')
cache_path = '../cache1/'
flag = 0
sub = 1
the_data = sys.argv[2]
def make_feat(args):
	if not os.path.exists('../cache/train_pd_feat6.hdf') or 1:
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
		train_pd = train_pd.append(test_pd).reset_index(drop=True)
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
		adFeature_pd[one_hot_aid_feature] = adFeature_pd[one_hot_aid_feature].astype('uint8')
		adFeature_pd['aid'] = adFeature_pd['aid'].astype('int16')
		
		# uid
		num_feat = ['LBS', 'age', 'carrier', 'consumptionAbility', 'education', 'gender', 'house', 'os', 'ct',
		            'marriageStatus']
		# str_feat = ['interest1','interest2','interest3','interest4','interest5','kw1','kw2','kw3','topic1','topic2','topic3','appIdInstall','appIdAction']
		
		userFeature_pd = pd.read_csv('../input/'+the_data+'/userFeature.csv', dtype=dtype_list,
		                             usecols=num_feat + ['uid'])
		userFeature_pd['gender'] = userFeature_pd['gender'].fillna(1).astype('int8')
		print(userFeature_pd.max())
		userFeature_pd['ct'] = userFeature_pd['ct'].fillna('-1').map(sort2str)
		userFeature_pd['marriageStatus'] = userFeature_pd['marriageStatus'].fillna('-1').map(sort2str)
		userFeature_pd['os'] = userFeature_pd['os'].fillna('-1').map(sort2str)
		userFeature_pd['LBS'] = userFeature_pd['LBS'].fillna(-1).astype('int16')
		userFeature_pd['house'] = userFeature_pd['house'].fillna(-1).astype('int8')
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
		train_pd.to_hdf('../cache/train_pd_feat6.hdf', 'w')
	else:
		train_pd = pd.read_hdf('../cache/train_pd_feat6.hdf', 'w')
	train_pd_columns = train_pd.columns.tolist()
	
	if args == '1' or sub :
		str_download_feat_hdf(train_pd, 0)
		train_pd = train_pd[train_pd_columns]
		gc.collect()
	if args == '10' or sub :
		decomposition_str_feat_hdf(train_pd, 0)
		train_pd = train_pd[train_pd_columns]
		gc.collect()
	if args == '11' or sub :
		str_aid_decomposition_feat_hdf(train_pd, 0)
		train_pd = train_pd[train_pd_columns]
		gc.collect()
	
	if args == '2' or sub :
		str_w2v_feat_hdf(train_pd, 0)
		train_pd = train_pd[train_pd_columns]
	
	if args == '3' or sub :
		str_len_feat_hdf(train_pd, 0)
		train_pd = train_pd[train_pd_columns]
		gc.collect()
	
	# str_label_feat_all_hdf(train_pd, 0)
	# train_pd = train_pd[train_pd_columns]
	# gc.collect()
	
	if args == '4' or sub :
		str_d2v_feat_hdf(train_pd, 0)
		train_pd = train_pd[train_pd_columns]
		gc.collect()
	
	if args == '5' or sub :
		str_uid_aid_decomposition_feat_hdf(train_pd, 0)
		train_pd = train_pd[train_pd_columns]
		gc.collect()
		
		str_aid_uid_decomposition_feat_hdf(train_pd, 0)
		train_pd = train_pd[train_pd_columns]
	
	if args == '6' or sub :
		print('kaishi')
		str_uid_decomposition_feat_hdf_fuck(train_pd, 0)
		train_pd = train_pd[train_pd_columns]
		gc.collect()
	
	if args == '7' or sub :
		decomposition_str_feat_hdf_fuck(train_pd, 0)
		train_pd = train_pd[train_pd_columns]
	
	if args == '8' or sub :
		str_uid_uid_aid_decomposition_feat_hdf_fuck(train_pd, 0)
		train_pd = train_pd[train_pd_columns]
	
	if args == '9' or sub :
		str_download_feat_mean_hdf(train_pd, 0)
		train_pd = train_pd[train_pd_columns]
	
	if args == '20' or sub :
		topn_mean_mean(train_pd, 1)
		train_pd = train_pd[train_pd_columns]
	
	if args == '21' or sub :
		topn_mean_mean(train_pd, 2)
		train_pd = train_pd[train_pd_columns]
	
	if args == '22' or sub :
		topn_mean_mean(train_pd, 3)
		train_pd = train_pd[train_pd_columns]
		
	if args == '23' or sub :
		split_str_feat(train_pd, 0)
		train_pd = train_pd[train_pd_columns]
	
	if args == '999' or sub :
		str_label_feat_all_hdf(train_pd, 0)
	
	if args == '0':
		str_download_feat_hdf(train_pd, 0)
		train_pd = train_pd[train_pd_columns]
		
		decomposition_str_feat_hdf(train_pd, 0)
		train_pd = train_pd[train_pd_columns]
		
		str_aid_decomposition_feat_hdf(train_pd, 0)
		train_pd = train_pd[train_pd_columns]
		
		str_w2v_feat_hdf(train_pd, 0)
		train_pd = train_pd[train_pd_columns]
		
		str_len_feat_hdf(train_pd, 0)
		train_pd = train_pd[train_pd_columns]
		
		str_label_feat_all_hdf(train_pd, 0)
		train_pd = train_pd[train_pd_columns]
		
		str_d2v_feat_hdf(train_pd, 0)
		train_pd = train_pd[train_pd_columns]
		
		str_uid_aid_decomposition_feat_hdf(train_pd, 0)
		train_pd = train_pd[train_pd_columns]
		str_aid_uid_decomposition_feat_hdf(train_pd, 0)
		train_pd = train_pd[train_pd_columns]


a = sys.argv[1]
make_feat(a)