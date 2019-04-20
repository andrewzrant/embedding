# coding=utf-8
import pandas as pd
import numpy as  np
import os
import gc

import warnings
from gensim.models import Word2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.decomposition import LatentDirichletAllocation, NMF, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
from scipy.stats import entropy
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")

cache_path = '../cache1/'
flag = 0

a = open('datas.txt','r')
the_data =a.readline()



def concat(L):
	result = None
	for l in L:
		if result is None:
			result = l
		else:
			try:
				result[l.columns.tolist()] = l
			except:
				result[l.name] = l
	return result


def cache(func):
	def wrapper(*args, **kwargs):
		funcName = func.__name__
		the_size = '_' + str(args[0].shape[0])
		if len(args) > 1 and len(str(args[1])) > 30:
			canshu = '_'.join([str(x) for x in args[2:]])
		else:
			canshu = '_'.join([str(x) for x in args[1:]])
		file_name = cache_path + funcName + the_size + '_' + canshu + '_cache' + '.hdf'
		if os.path.exists(file_name) and flag:  # and 'get_label_feat' not in funcName:
			print("load : " + funcName)
			result = pd.read_hdf(file_name, 'w')
			gc.collect()
			return result
		else:
			print("make : " + funcName)
			result = func(*args, **kwargs)
			result.to_hdf(file_name, 'w')
			gc.collect()
			return result
	
	return wrapper


def sort2str(x):
	x = x.split(' ')
	x = sorted(x, key=lambda x: int(x))
	return ' '.join(x)


def download_feat_xuan(data, feat_set, fe):
	use_feat = fe.split('_')
	the_feat = feat_set.groupby(use_feat, as_index=False)['label'].agg({
		'the_sum': 'sum', 'the_mean': 'mean'
	})
	data = data.merge(the_feat, 'left', use_feat).fillna(-1)
	return data['the_sum'], data['the_mean']


def each_down_load_feat(data, fe, cv_num, f):
	use_feat = fe.split('_') + ['label']
	data = data[use_feat]
	feat_set = data[data['label'] != -1]
	result = np.zeros((feat_set.shape[0], 1))
	result_1 = np.zeros((feat_set.shape[0], 1))
	kf = StratifiedKFold(n_splits=cv_num, shuffle=True, random_state=520).split(feat_set, feat_set['label'])
	for k, (train_fold, test_fold) in enumerate(kf):
		result[test_fold, 0], result_1[test_fold, 0] = f(feat_set.loc[test_fold, :].copy(),
		                                                 feat_set.loc[train_fold, :].copy(), fe)
	label_fold, label_fold_1 = f(data[data['label'] == -1].copy(), feat_set.copy(), fe)
	label_fold = [1.0 * i / cv_num * (cv_num - 1) for i in label_fold.values]
	
	data['_1'] = [x[0] for x in list(result)] + list(label_fold)
	data['_2'] = [x[0] for x in list(result_1)] + list(label_fold_1)
	return data['_1'], data['_2']


def str_download_feat(data, feat_set, str_fe, fe):
	all_dict = {}
	pos_dict = {}
	for fe_value, each in feat_set.groupby([fe]):
		all_word = list(each[str_fe].map(
			lambda x: x.split(' ')).values
		                )
		each_1 = each[each['label'] == 1]
		pos_word = list(each_1[str_fe].map(
			lambda x: x.split(' ')).values
		                )
		all_word_dict = {}
		for each_line in all_word:
			for each in each_line:
				if each in all_word_dict:
					all_word_dict[each] += 1
				else:
					all_word_dict[each] = 1
		pos_word_dict = {}
		for each_line in pos_word:
			for each in each_line:
				if each in pos_word_dict:
					pos_word_dict[each] += 1
				else:
					pos_word_dict[each] = 1
		all_dict[fe_value] = all_word_dict
		pos_dict[fe_value] = pos_word_dict
	data['str'] = data[str_fe].map(
		lambda x: str(x).split(' ')
	)
	
	
	def zuida(str_list, fe):
		max_values = -1
		for i in range(len(str_list)):
			pos_count = pos_dict[fe].get(str_list[i], 0)
			all_count = all_dict[fe].get(str_list[i], 0)
			if all_count > 0 and pos_count * 1.0 / all_count > max_values:
				max_values = pos_count * 1.0 / all_count
		return max_values
	
	def sum_jun(str_list, fe):
		pos_sum = []
		all_sum = []
		for i in range(len(str_list)):
			pos_count = pos_dict[fe].get(str_list[i], 0)
			all_count = all_dict[fe].get(str_list[i], 0)
			if all_count > 0:
				pos_sum.append(pos_count)
				all_sum.append(all_count)
		if len(all_sum) == 0:
			return 0
		else:
			return sum(pos_sum) / sum(all_sum)
	
	def jun(str_list, fe):
		result = []
		for i in range(len(str_list)):
			pos_count = pos_dict[fe].get(str_list[i], 0)
			all_count = all_dict[fe].get(str_list[i], 0)
			if all_count > 0:
				result.append(pos_count * 1.0 / all_count)
		if len(result) == 0:
			return 0
		else:
			return sum(result) / len(result)
	
	def num_f(str_list, fe):
		result = 0
		lens = 0
		for i in range(len(str_list)):
			result += pos_dict[fe].get(str_list[i], 0)
			if str_list[i] in pos_dict[fe]:
				lens += 1
		if lens > 0:
			return result / lens
		else:
			return 0
	
	max_ = list(map(lambda x, y: zuida(x, y), data['str'], data[fe]))
	mean_ = list(map(lambda x, y: jun(x, y), data['str'], data[fe]))
	num_ = list(map(lambda x, y: num_f(x, y), data['str'], data[fe]))
	sum_mean_ = list(map(lambda x, y: sum_jun(x, y), data['str'], data[fe]))
	return max_, mean_, num_, sum_mean_


def str_download_feats_(data, str_fe, fe, cv_num, f):
	data['new_fe'] = jiaocha(data, fe)
	data = data[['uid', 'label', 'new_fe', str_fe]]
	
	feat_set = data[data['label'] != -1]
	result_0 = np.zeros((feat_set.shape[0], 1))
	result_1 = np.zeros((feat_set.shape[0], 1))
	result_2 = np.zeros((feat_set.shape[0], 1))
	result_3 = np.zeros((feat_set.shape[0], 1))
	kf = StratifiedKFold(n_splits=cv_num,
	                     shuffle=True,
	                     random_state=520).split(feat_set, feat_set['label'])
	for k, (train_fold, test_fold) in enumerate(kf):
		result_0[test_fold, 0], result_1[test_fold, 0], result_2[test_fold, 0], result_3[test_fold, 0] \
			= f(
			feat_set.loc[test_fold, :].copy(), feat_set.loc[train_fold, :].copy(), str_fe, 'new_fe'
		)
	label_fold_0, label_fold_1, label_fold_2, label_fold_3 = f(
		data[data['label'] == -1].copy(), feat_set.copy(),str_fe, 'new_fe')
	label_fold_2 = [1.0 * i / cv_num * (cv_num - 1) for i in label_fold_2]
	
	result_0 = [x[0] for x in list(result_0)] + list(label_fold_0)
	result_1 = [x[0] for x in list(result_1)] + list(label_fold_1)
	result_2 = [x[0] for x in list(result_2)] + list(label_fold_2)
	result_3 = [x[0] for x in list(result_3)] + list(label_fold_3)
	return result_0, result_1, result_2, result_3


def jiaocha(data, feat_list):
	data = data[feat_list]
	name = '_'.join(feat_list) + '_add'
	data[name] = 0
	last_max = 0
	for each in feat_list:
		data[name] = data[name] * last_max + data[each]
	return data[name]


def entropy_feat(data, feat_list1, feat_list2):
	feat1 = '_'.join(feat_list1) + '_add'
	feat2 = '_'.join(feat_list2) + '_add'
	data = data[feat_list1 + feat_list2]
	data[feat1] = jiaocha(data, feat_list1)
	data[feat2] = jiaocha(data, feat_list2)
	name = feat1 + '_' + feat2 + '_'
	cc = np.zeros((int(data[feat1].max() + 1), int(data[feat2].max() + 1)), dtype=np.int32)
	np.add.at(cc, (data[feat1].values, data[feat2].values), 1)
	max_rate = cc.max(1) / cc.sum(1)
	max_value = cc.argmax(1)
	nunique = (cc > 0).sum(1)
	entropy_feat = list(map(entropy, cc))
	the_feat = np.concatenate(([max_rate], [max_value], [nunique], [entropy_feat]))
	the_feat = pd.DataFrame(the_feat.T,
	                        columns=[name + 'max_rate', name + 'max_value', name + 'nunique', name + 'entropy'])
	the_feat[feat1] = [i for i in range(the_feat.shape[0])]
	result = data[[feat1]].merge(the_feat, on=[feat1], how='left').fillna(-1)
	del result[feat1]
	return result[name + 'max_rate'], result[name + 'max_value'], result[name + 'nunique'], result[name + 'entropy']


def get_count_rate(data, feat_set, fe, fe1):
	data = data[list(set(fe + fe1))]
	name = "_".join(fe) + '_by_' + "_".join(fe1)
	fe_count = feat_set.groupby(fe, as_index=False)[fe[0]].agg({name + '_fe': 'count'})
	fe1_count = feat_set.groupby(fe1, as_index=False)[fe1[0]].agg({name + '_fe1': 'count'})
	data = data.merge(fe_count, on=fe, how='left').fillna(0)
	data = data.merge(fe1_count, on=fe1, how='left').fillna(0)
	data[name + '_rate'] = (data[name + '_fe1'] + 0.001) / (data[name + '_fe'] + 0.001)
	return data[name + '_rate']


@cache
def decomposition_cross_feat(data, feat1, feat2, n_lda, n_nmf, n_svd):
	data = data[[feat1] + feat2]
	if len(feat2) != 1:
		new_feat = data[feat2].fillna(-1).values
		new_feat_list = []
		for each_line in new_feat:
			strs = ''
			for i, each in enumerate(each_line):
				for fuck in str(each).split(' '):
					strs = strs + ' ' + str(i) + '_' + fuck
			new_feat_list.append(strs)
		feat2 = '_'.join(feat2)
		data[feat2] = new_feat_list
	else:
		feat2 = feat2[0]
		data[feat2] = data[feat2].map(lambda x: ' '.join([i + 'a' for i in str(x).split(' ')]))
	
	feat_dict = {}
	for each in data[[feat1, feat2]].values:
		feat_dict.setdefault(each[0], []).append(str(each[1]))
	feat_list = list(feat_dict.keys())
	feat_as_sentence = [' '.join(feat_dict[each]) for each in feat_list]
	feat_as_matrix = CountVectorizer(token_pattern='(?u)\\b\\w+\\b').fit_transform(feat_as_sentence)
	mycolumns = []
	data_list = []
	print('start')
	if n_lda > 0:
		lda = LatentDirichletAllocation(n_components=n_lda, n_jobs=-1, batch_size=data.shape[0] + 1,
		                                random_state=1)
		lda_feat = lda.fit_transform(feat_as_matrix)
		feat1_lda_pd = pd.DataFrame(lda_feat,
		                            columns=['lda_num_cross_' + feat1 + '_by_' + feat2 + str(i) for i in range(n_lda)])
		mycolumns += ['lda_num_cross_' + feat1 + '_by_' + feat2 + str(i) for i in range(n_lda)]
		data_list.append(feat1_lda_pd)
	
	if n_nmf > 0:
		nmf = NMF(n_components=n_nmf, random_state=1)
		nmf_feat = nmf.fit_transform(feat_as_matrix)
		feat1_nmf_pd = pd.DataFrame(nmf_feat,
		                            columns=['nmf_num_cross_' + feat1 + '_by_' + feat2 + str(i) for i in range(n_nmf)])
		mycolumns += ['nmf_num_cross_' + feat1 + '_by_' + feat2 + str(i) for i in range(n_nmf)]
		data_list.append(feat1_nmf_pd)
	
	if n_svd > 0:
		svd = TruncatedSVD(n_components=n_svd)
		svd_feat = svd.fit_transform(feat_as_matrix)
		feat1_svd_pd = pd.DataFrame(svd_feat,
		                            columns=['svd_num_cross_' + feat1 + '_by_' + feat2 + str(i) for i in range(n_svd)])
		mycolumns += ['svd_num_cross_' + feat1 + '_by_' + feat2 + str(i) for i in range(n_svd)]
		data_list.append(feat1_svd_pd)
	print('end')
	the_feat = concat(data_list)
	the_feat[feat1] = feat_list
	
	data = data[[feat1, feat2]].fillna(-1).merge(the_feat, on=[feat1], how='left')
	return data[mycolumns].astype('float16')


@cache
def str_decomposition_cross_feat(data, feat1, feat2, n_lda, n_nmf, n_svd):
	data = data[[feat1] + feat2]
	if len(feat2) != 1:
		new_feat = data[feat2].fillna(-1).values
		new_feat_list = []
		for each_line in new_feat:
			strs = ''
			for i, each in enumerate(each_line):
				for fuck in str(each).split(' '):
					strs = strs + ' ' + str(i) + '_' + fuck
			new_feat_list.append(strs)
		feat2 = '_'.join(feat2)
		data[feat2] = new_feat_list
	else:
		feat2 = feat2[0]
		data[feat2] = data[feat2].map(lambda x: ' '.join([i + 'a' for i in str(x).split(' ')]))
	
	data[feat1] = data[feat1].map(lambda x: str(x))
	feat_dict = {}
	for each in data[[feat1, feat2]].values:
		for each_0 in each[0].split(' '):
			feat_dict.setdefault(each_0, []).append(str(each[1]))
	
	feat_list = list(feat_dict.keys())
	feat_as_sentence = [' '.join(feat_dict[each]) for each in feat_list]
	feat_as_matrix = CountVectorizer(token_pattern='(?u)\\b\\w+\\b').fit_transform(feat_as_sentence)
	print('start')
	mycolumns = []
	data_list = []
	if n_lda > 0:
		lda = LatentDirichletAllocation(n_components=n_lda, n_jobs=-1, batch_size=data.shape[0] + 1,
		                                random_state=1)
		lda_feat = lda.fit_transform(feat_as_matrix)
		feat1_lda_pd = pd.DataFrame(lda_feat,
		                            columns=['lda_str_cross_' + feat1 + '_by_' + feat2 + str(i) for i in range(n_lda)])
		mycolumns += ['lda_str_cross_' + feat1 + '_by_' + feat2 + str(i) for i in range(n_lda)]
		data_list.append(feat1_lda_pd)
	
	if n_nmf > 0:
		nmf = NMF(n_components=n_nmf, random_state=1)
		nmf_feat = nmf.fit_transform(feat_as_matrix)
		feat1_nmf_pd = pd.DataFrame(nmf_feat,
		                            columns=['nmf_str_cross_' + feat1 + '_by_' + feat2 + str(i) for i in range(n_nmf)])
		mycolumns += ['nmf_str_cross_' + feat1 + '_by_' + feat2 + str(i) for i in range(n_nmf)]
		data_list.append(feat1_nmf_pd)
	
	if n_svd > 0:
		svd = TruncatedSVD(n_components=n_svd)
		svd_feat = svd.fit_transform(feat_as_matrix)
		feat1_svd_pd = pd.DataFrame(svd_feat,
		                            columns=['svd_str_cross_' + feat1 + '_by_' + feat2 + str(i) for i in range(n_svd)])
		mycolumns += ['svd_str_cross_' + feat1 + '_by_' + feat2 + str(i) for i in range(n_svd)]
		data_list.append(feat1_svd_pd)
	print('end')
	the_feat = concat(data_list)
	the_feat[feat1] = feat_list
	feat_list = the_feat.values.tolist()
	feat_dict = {}
	for each in feat_list:
		feat_dict[each[-1]] = each[:-1]
	feat1_list = data[feat1].values
	rusult = []
	for each in feat1_list:
		each_id = each.split(' ')
		each_result = [0, ] * len(feat_dict[each_id[0]])
		for id in each_id:
			for i, i_value in enumerate(feat_dict[id]):
				each_result[i] += i_value
		each_result = [i / len(each_id) for i in each_result]
		rusult.append(each_result)
	
	rusult = pd.DataFrame(rusult, columns=mycolumns)
	return rusult.astype('float16')


def str2w2v_mean(x, model, victor_size):
	result = []
	for each in str(x).split(' '):
		if each in model:
			result.append(model[each])
	if len(result) == 0:
		return [0] * victor_size
	the_mean = []
	for i in range(victor_size):
		sum = 0
		num = 0
		for each in result:
			sum += each[i]
			num += 1
		the_mean.append(sum / num)
	return the_mean


@cache
def w2v_feat(data, fe, maxlen, victor_size):
	data = data[[fe]]
	model = Word2Vec([[word for word in document.split(' ')] for document in data[fe].map(lambda x: str(x)).values],
	                 size=victor_size, window=maxlen, iter=8, seed=2018, min_count=3)
	w2v_mean = data[fe].map(lambda x: str2w2v_mean(x, model, victor_size))
	w2v_mean = pd.DataFrame(list(w2v_mean), columns=[fe + '_w2v_mean_' + str(i) for i in range(victor_size)])
	return w2v_mean.astype('float16')


class LabeledLineSentence(object):
	def __init__(self, text, labels):
		self.text = text
		self.labels = labels
	
	def __iter__(self):
		length = self.text.shape[0]
		for i in range(length):
			yield TaggedDocument(words=self.text[i].split(' '), tags=self.labels[i])


@cache
def d2v_feat(data, fe, victor_size):
	print('d2v : ' + fe)
	sentences = LabeledLineSentence(text=data[fe].map(lambda x: str(x)).values,
	                                labels=[str(i) for i in range(data.shape[0])])
	model = Doc2Vec(
		dm=1,
		size=victor_size,
		negative=5,
		hs=0,
		min_count=1,
		window=5,
		sample=1e-5,
		workers=20,
		alpha=0.025,
		min_alpha=0.025,
	)
	model.build_vocab(sentences)
	model.train(documents=sentences, total_examples=model.corpus_count, epochs=6)
	d2v_value = data[fe].map(lambda x: model.infer_vector(str(x).split(' ')))
	
	d2v_value = pd.DataFrame(list(d2v_value), columns=[fe + '_d2v_' + str(i) for i in range(victor_size)])
	return d2v_value.astype('float16')


def get_second_add(data, key_word, no_word=''):
	col_list = []
	print(key_word)
	for each_col in list(data.columns):
		flag = 0
		for each_key in key_word:
			if each_key not in each_col:
				flag = 1
		for each_key in no_word:
			if each_key in no_word:
				flag = 1
		if flag == 0:
			col_list.append(each_col)
	name = '_'.join(key_word) + '_s_' + '_'.join(no_word) + '_second_add'
	data[name] = 0
	for each in col_list:
		data.ix[data[each] == -1, [each]] = 0
		data[each].fillna(0)
		data[name] = data[name] + data[each]
	data[name] = data[name].astype('float16')


def str_download_feat_hdf(train_pd, caches):
	if os.path.exists('../cache/str_download.hdf') and caches:
		print('load:  ../cache/str_download.hdf')
	else:
		str_feat = ['interest1', 'interest2', 'interest3', 'interest4', 'interest5', 'kw1', 'kw2', 'kw3', 'topic1',
		            'topic2', 'topic3', 'appIdInstall', 'appIdAction']
		for each in str_feat:
			print('download: ', each)
			a = open('datas.txt', 'r')
			the_data = a.readline()
			str_feat_set = pd.read_csv('../input/'+the_data+'/userFeature.csv', usecols=[each, 'uid'])
			train_pd = train_pd.merge(str_feat_set, 'left', ['uid']).fillna('-1')
			del str_feat_set
			gc.collect()
			train_pd[each + '_aid_max_download'], train_pd[each + '_aid_mean_download'], \
			train_pd[each + '_aid_num_download'], _ = str_download_feats_(
				train_pd, each, ['aid'], 5, str_download_feat)
			train_pd[each + '_aid_max_download'] = train_pd[each + '_aid_max_download'].astype('float16')
			train_pd[each + '_aid_mean_download'] = train_pd[each + '_aid_mean_download'].astype('float16')
			train_pd[each + '_aid_num_download'] = train_pd[each + '_aid_num_download'].astype('float32')
			# train_pd[each + '_aid_sum_mean_download'] = train_pd[each + '_aid_sum_mean_download'].astype('float16')
			gc.collect()
			train_pd['fuzhu'] = 1
			train_pd[each + '_max_download'], train_pd[each + '_mean_download'], \
			train_pd[each + '_num_download'], _ = str_download_feats_(
				train_pd, each, ['fuzhu'], 5, str_download_feat)
			train_pd[each + '_max_download'] = train_pd[each + '_max_download'].astype('float16')
			train_pd[each + '_mean_download'] = train_pd[each + '_mean_download'].astype('float16')
			train_pd[each + '_num_download'] = train_pd[each + '_num_download'].astype('float32')
			# train_pd[each + '_sum_mean_download'] = train_pd[each + '_sum_mean_download'].astype('float16')
			del train_pd[each]
			gc.collect()
			download_list = [x for x in train_pd.columns if '_download' in x and each in x]
			train_pd[download_list].to_hdf('../cache/' + each + 'str_download.hdf', 'w')
		download_list = [x for x in train_pd.columns if '_download' in x]
		train_pd[download_list].to_hdf('../cache/str_download.hdf', 'w')
	print('download')


def str_download_feat_mean_hdf(train_pd, caches):
	download_feats = pd.read_hdf('../cache/str_download.hdf', 'w')
	download_list = [x for x in download_feats.columns if '_aid' not in x and '_mean' in x]
	print(download_list)
	train_pd = concat([train_pd, download_feats])
	for each in download_list:
		temp = train_pd.groupby(['aid'], as_index=False)[each].agg({each + '_means': 'mean'})
		train_pd = train_pd.merge(temp, 'left', ['aid']).fillna(0)
		train_pd[each + '_means'] = train_pd[each + '_means'].astype('float16')
		
		temp = train_pd.groupby(['uid'], as_index=False)[each].agg({each + '_means_mean': 'mean'})
		train_pd = train_pd.merge(temp, 'left', ['uid']).fillna(0)
		train_pd[each + '_means_mean'] = train_pd[each + '_means_mean'].astype('float16')
	download_list = [x for x in train_pd.columns if '_means' in x]
	print(download_list)
	train_pd[download_list].to_hdf('../cache/str_download_means.hdf', 'w')


def decomposition_str_feat_hdf(train_pd, caches):
	if os.path.exists('../cache/str_decomposition.hdf') and caches:
		print('load:  ../cache/str_decomposition.hdf')
	else:
		str_feat = ['interest1', 'interest2', 'interest5', 'kw1', 'kw2', 'topic1', 'topic2']
		for each in str_feat:
			print('decomposition_str: ', each)
			a = open('datas.txt', 'r')
			the_data = a.readline()
			str_feat_set = pd.read_csv('../'+the_data+'/data/userFeature.csv', usecols=[each, 'uid'])
			train_pd = train_pd.merge(str_feat_set, 'left', ['uid']).fillna('-1')
			del str_feat_set
			gc.collect()
			temp = str_decomposition_cross_feat(train_pd, each, ['aid'], 0, 5, 5)
			train_pd = concat([train_pd, temp])
			print(train_pd.columns)
			for i in range(5):
				temp = train_pd.groupby(['aid'], as_index=False)['svd_str_cross_' + each + '_by_aid' + str(i)].agg(
					{'svd_str_cross_' + each + '_by_aid' + str(i) + '_uid_mean': 'mean'})
				train_pd = train_pd.merge(temp, 'left', ['aid']).fillna(0)
				train_pd['svd_str_cross_' + each + '_by_aid' + str(i) + '_uid_mean'] = train_pd[
					'svd_str_cross_' + each + '_by_aid' + str(i) + '_uid_mean'].astype('float16')
			del train_pd[each]
		
		download_list = [x for x in train_pd.columns if 'str_cross' in x]
		train_pd[download_list].to_hdf('../cache/str_decomposition.hdf', 'w')
		print(download_list)
	print('decomposition_str')


def str_aid_decomposition_feat_hdf(train_pd, caches):
	if os.path.exists('../cache/str_aid_decomposition.hdf') and caches:
		print('load:  ../cache/str_aid_decomposition.hdf')
	else:
		str_feat = ['interest1', 'interest2', 'interest5', 'kw1', 'kw2', 'topic1', 'topic2']
		for each in str_feat:
			print('str_aid_decomposition: ', each)
			a = open('datas.txt', 'r')
			the_data = a.readline()
			str_feat_set = pd.read_csv('../input/'+the_data+'/userFeature.csv', usecols=[each, 'uid'])
			train_pd = train_pd.merge(str_feat_set, 'left', ['uid']).fillna('-1')
			del str_feat_set
			gc.collect()
			temp = decomposition_cross_feat(train_pd, 'aid', [each], 0, 5, 5)
			train_pd = concat([train_pd, temp])
			print(train_pd.columns)
			for i in range(5):
				temp = train_pd.groupby(['uid'], as_index=False)['svd_num_cross_aid_by_' + each + str(i)].agg(
					{'svd_num_cross_aid_by_' + each + str(i) + '_uid_mean': 'mean'})
				train_pd = train_pd.merge(temp, 'left', ['uid']).fillna(0)
				train_pd['svd_num_cross_aid_by_' + each + str(i) + '_uid_mean'] = train_pd[
					'svd_num_cross_aid_by_' + each + str(i) + '_uid_mean'].astype('float16')
			del train_pd[each]
		
		download_list = [x for x in train_pd.columns if 'num_cross' in x]
		print(download_list)
		train_pd[download_list].to_hdf('../cache/str_aid_decomposition.hdf', 'w')
	print('str_aid_decomposition')


# ================================================================================
def decomposition_str_feat_hdf_fuck(train_pd, caches):
	if os.path.exists('../cache/str_decomposition_fuck.hdf') and caches:
		print('load:  ../cache/str_decomposition_fuck.hdf')
	else:
		str_feat = ['interest1', 'interest2', 'interest5', 'kw1', 'kw2', 'topic1', 'topic2']
		for each in str_feat:
			print('decomposition_str: ', each)
			a = open('datas.txt', 'r')
			the_data = a.readline()
			str_feat_set = pd.read_csv('../input/'+the_data+'/userFeature.csv', usecols=[each, 'uid'])
			train_pd = train_pd.merge(str_feat_set, 'left', ['uid']).fillna('-1')
			del str_feat_set
			gc.collect()
			temp = str_decomposition_cross_feat(train_pd, each, ['uid'], 0, 5, 5)
			train_pd = concat([train_pd, temp])
			print(train_pd.columns)
			for i in range(5):
				temp = train_pd.groupby(['aid'], as_index=False)['svd_str_cross_' + each + '_by_uid' + str(i)].agg(
					{'svd_str_cross_' + each + '_by_uid' + str(i) + '_aid_junuck': 'mean'})
				train_pd = train_pd.merge(temp, 'left', ['aid']).fillna(0)
				train_pd['svd_str_cross_' + each + '_by_uid' + str(i) + '_aid_junuck'] = train_pd[
					'svd_str_cross_' + each + '_by_uid' + str(i) + '_aid_junuck'].astype('float16')
			del train_pd[each]
		
		download_list = [x for x in train_pd.columns if 'str_cross' in x]
		train_pd[download_list].to_hdf('../cache/str_decomposition_fuck.hdf', 'w')
		print(download_list)
	print('decomposition_str')


def str_uid_decomposition_feat_hdf_fuck(train_pd, caches):
	if os.path.exists('../cache/str_uid_decomposition_fuck.hdf') and caches:
		print('load:  ../cache/str_uid_decomposition_fuck.hdf')
	else:
		str_feat = ['interest1', 'interest2', 'interest5', 'kw1', 'kw2', 'topic1', 'topic2']
		for each in str_feat:
			print('str_aid_decomposition: ', each)
			a = open('datas.txt', 'r')
			the_data = a.readline()
			str_feat_set = pd.read_csv('../input'+the_data+'/userFeature.csv', usecols=[each, 'uid'])
			train_pd = train_pd.merge(str_feat_set, 'left', ['uid']).fillna('-1')
			del str_feat_set
			gc.collect()
			temp = decomposition_cross_feat(train_pd, 'uid', [each], 0, 5, 5)
			train_pd = concat([train_pd, temp])
			print(train_pd.columns)
			del train_pd[each]
		
		download_list = [x for x in train_pd.columns if 'num_cross' in x]
		print(download_list)
		train_pd[download_list].to_hdf('../cache/str_uid_decomposition_fuck.hdf', 'w')
	print('str_aid_decomposition')


def str_uid_uid_aid_decomposition_feat_hdf_fuck(train_pd, caches):
	if os.path.exists('../cache/str_uid_uid_aid_decomposition.hdf') and caches:
		print('load:  ../cache/str_uid_uid_aid_decomposition.hdf')
	else:
		
		temp = str_decomposition_cross_feat(train_pd, 'uid', ['aid'], 0, 3, 3)
		temp.to_hdf('../cache/str_uid_uid_aid_decomposition.hdf', 'w')
		temp = str_decomposition_cross_feat(train_pd, 'aid', ['uid'], 0, 3, 3)
		temp.to_hdf('../cache/str_aid_uid_aid_decomposition.hdf', 'w')
	print('str_aid_decomposition')


# ==============================================================================================





def str_aid_uid_decomposition_feat_hdf(train_pd, caches):
	if os.path.exists('../cache/str_aid_uid_decomposition.hdf') and caches:
		print('load:  ../cache/str_aid_uid_decomposition.hdf')
	else:
		uid_col = ['LBS', 'age', 'carrier', 'consumptionAbility', 'education', 'gender', 'house', 'os', 'ct',
		           'marriageStatus']
		temp = decomposition_cross_feat(train_pd, 'aid', uid_col, 0, 5, 5)
		temp.columns = ['svd_num_cross_aid_by_uidfeat' + str(i) for i in range(5)]
		train_pd = concat([train_pd, temp])
		print(train_pd.columns)
		for i in range(5):
			temp = train_pd.groupby(['uid'], as_index=False)['svd_num_cross_aid_by_' + 'uidfeat' + str(i)].agg(
				{'svd_num_cross_aid_by_' + 'uidfeat' + str(i) + '_uid_mean': 'mean'})
			train_pd = train_pd.merge(temp, 'left', ['uid']).fillna(0)
			train_pd['svd_num_cross_aid_by_' + 'uidfeat' + str(i) + '_uid_mean'] = train_pd[
				'svd_num_cross_aid_by_' + 'uidfeat' + str(i) + '_uid_mean'].astype('float16')
		
		download_list = [x for x in train_pd.columns if 'uidfeat' in x]
		train_pd[download_list].to_hdf('../cache/str_aid_uid_decomposition.hdf', 'w')
		print(download_list)
	print('str_aid_decomposition')


def str_uid_aid_decomposition_feat_hdf(train_pd, caches):
	if os.path.exists('../cache/str_uid_aid_decomposition_feat.hdf') and caches:
		print('load:  ../cache/str_uid_aid_decomposition_feat.hdf')
	else:
		uid_col = ['LBS', 'age', 'carrier', 'consumptionAbility', 'education', 'gender', 'house', 'os', 'ct',
		           'marriageStatus']
		temp = decomposition_cross_feat(train_pd, 'uid', uid_col, 0, 5, 5)
		temp.columns = ['svd_num_cross_uid_by_uidfeat' + str(i) for i in range(5)]
		train_pd = concat([train_pd, temp])
		print(train_pd.columns)
		for i in range(5):
			temp = train_pd.groupby(['aid'], as_index=False)['svd_num_cross_uid_by_uidfeat' + str(i)].agg(
				{'svd_num_cross_uid_by_uidfeat' + str(i) + '_aid_mean': 'mean'})
			train_pd = train_pd.merge(temp, 'left', ['aid']).fillna(0)
			train_pd['svd_num_cross_uid_by_uidfeat' + str(i) + '_aid_mean'] = train_pd[
				'svd_num_cross_uid_by_uidfeat' + str(i) + '_aid_mean'].astype('float16')
			
			temp = train_pd.groupby(['uid'], as_index=False)[
				'svd_num_cross_uid_by_' + 'uidfeat' + str(i) + '_aid_mean'].agg(
				{'svd_num_cross_uid_by_' + 'uidfeat' + str(i) + '_aid_uid_mean': 'mean'})
			train_pd = train_pd.merge(temp, 'left', ['uid']).fillna(0)
			train_pd['svd_num_cross_uid_by_' + 'uidfeat' + str(i) + '_aid_uid_mean'] = train_pd[
				'svd_num_cross_uid_by_' + 'uidfeat' + str(i) + '_aid_uid_mean'].astype('float16')
		
		download_list = [x for x in train_pd.columns if 'uid_by_uidfeat' in x]
		print(download_list)
		train_pd[download_list].to_hdf('../cache/str_uid_aid_decomposition_feat.hdf', 'w')
	print('str_aid_decomposition')


def str_w2v_feat_hdf(train_pd, caches):
	if os.path.exists('../cache/str_w2v.hdf') and caches:
		print('load:  ../cache/str_w2v.hdf')
	else:
		str_feat = ['interest1', 'interest2', 'interest3', 'interest4', 'interest5', 'kw1', 'kw2', 'kw3', 'topic1',
		            'topic2']
		for each in str_feat:
			print('w2v: ', each)
			a = open('datas.txt', 'r')
			the_data = a.readline()
			str_feat_set = pd.read_csv('../input/'+the_data+'/userFeature.csv', usecols=[each, 'uid'])
			train_pd = train_pd.merge(str_feat_set, 'left', ['uid']).fillna('-1')
			del str_feat_set
			gc.collect()
			temp = w2v_feat(train_pd, each, 20, 5)
			train_pd = concat([train_pd, temp])
			for i in range(5):
				temp = train_pd.groupby(['aid'], as_index=False)[each + '_w2v_mean_' + str(i)].agg(
					{each + '_w2v_mean_' + str(i) + '_aid_mean': 'mean'})
				train_pd = train_pd.merge(temp, 'left', ['aid']).fillna(0)
				
				temp = train_pd.groupby(['uid'], as_index=False)[each + '_w2v_mean_' + str(i) + '_aid_mean'].agg(
					{each + '_w2v_mean_' + str(i) + '_aid_uid_mean': 'mean'})
				train_pd = train_pd.merge(temp, 'left', ['uid']).fillna(0)
				train_pd[each + '_w2v_mean_' + str(i) + '_aid_mean'] = train_pd[
					each + '_w2v_mean_' + str(i) + '_aid_mean'].astype('float16')
				train_pd[each + '_w2v_mean_' + str(i) + '_aid_uid_mean'] = train_pd[
					each + '_w2v_mean_' + str(i) + '_aid_uid_mean'].astype('float16')
			del train_pd[each]
		
		w2v_list = [x for x in train_pd.columns if 'w2v' in x]
		train_pd[w2v_list].to_hdf('../cache/str_w2v.hdf', 'w')
	print('w2v')


def str_d2v_feat_hdf(train_pd, caches):
	if os.path.exists('../cache/str_d2v.hdf') and caches:
		print('load:  ../cache/str_d2v.hdf')
	else:
		str_feat = ['interest1', 'interest2', 'interest3', 'interest4', 'interest5', 'kw1', 'kw2', 'kw3', 'topic1',
		            'topic2', 'topic3', 'appIdInstall', 'appIdAction']
		for each in str_feat:
			print('d2v: ', each)
			a = open('datas.txt', 'r')
			the_data = a.readline()
			str_feat_set = pd.read_csv('../input/'+the_data+'/userFeature.csv', usecols=[each, 'uid'])
			train_pd = train_pd.merge(str_feat_set, 'left', ['uid']).fillna('-1')
			del str_feat_set
			gc.collect()
			temp = d2v_feat(train_pd, each, 5)
			train_pd = concat([train_pd, temp])
			# for i in range(5):
			# 	temp = train_pd.groupby(['aid'], as_index=False)[each + '_d2v_mean_' + str(i)].agg(
			# 		{each + '_d2v_mean_' + str(i) + '_aid_mean': 'mean'})
			# 	train_pd = train_pd.merge(temp, 'left', ['aid']).fillna(0)
			#
			# 	temp = train_pd.groupby(['uid'], as_index=False)[each + '_d2v_mean_' + str(i) + '_aid_mean'].agg(
			# 		{each + '_d2v_mean_' + str(i) + '_aid_uid_mean': 'mean'})
			# 	train_pd = train_pd.merge(temp, 'left', ['uid']).fillna(0)
			# 	train_pd[each + '_d2v_mean_' + str(i) + '_aid_mean'] = train_pd[
			# 		each + '_d2v_mean_' + str(i) + '_aid_mean'].astype('float16')
			# 	train_pd[each + '_d2v_mean_' + str(i) + '_aid_uid_mean'] = train_pd[
			# 		each + '_d2v_mean_' + str(i) + '_aid_uid_mean'].astype('float16')
			del train_pd[each]
		
		d2v_list = [x for x in train_pd.columns if 'd2v' in x]
		train_pd[d2v_list].to_hdf('../cache/str_d2v.hdf', 'w')
	print('d2v')


def str_len_feat_hdf(train_pd, caches):
	if os.path.exists('../cache/str_len.hdf') and caches:
		print('load:  ../cache/str_len.hdf')
	else:
		str_feat = ['interest1', 'interest2', 'interest5', 'kw1', 'kw2', 'topic1', 'topic2', ]
		for each in str_feat:
			print('len: ', each)
			a = open('datas.txt', 'r')
			the_data = a.readline()
			str_feat_set = pd.read_csv('../input/'+the_data+'/userFeature.csv', usecols=[each, 'uid'])
			train_pd = train_pd.merge(str_feat_set, 'left', ['uid']).fillna('-1')
			del str_feat_set
			gc.collect()
			train_pd[each + 'len'] = train_pd[each].map(lambda x: 0 if x == '-1' else len(x.split(' ')))
			temp = train_pd.groupby(['aid'], as_index=False)[each + 'len'].agg({each + 'len' + '_aid_mean': 'mean'})
			train_pd = train_pd.merge(temp, 'left', ['aid']).fillna(0)
			temp = train_pd.groupby(['uid'], as_index=False)[each + 'len' + '_aid_mean'].agg(
				{each + 'len' + '_aid_uid_mean': 'mean'})
			train_pd = train_pd.merge(temp, 'left', ['uid']).fillna(0)
			
			_, train_pd[each + 'len_download_rate'] = each_down_load_feat(train_pd, each + 'len', 5, download_feat_xuan)
			_, train_pd[each + 'len_aid_download_rate'] = each_down_load_feat(train_pd, each + 'len', 5, download_feat_xuan)
			train_pd[each + 'len' + '_aid_mean'] = train_pd[each + 'len' + '_aid_mean'].astype('float16')
			train_pd[each + 'len' + '_aid_uid_mean'] = train_pd[each + 'len' + '_aid_uid_mean'].astype('float16')
			train_pd[each + 'len_download_rate'] = train_pd[each + 'len_download_rate'].astype('float16')
			train_pd[each + 'len_aid_download_rate'] = train_pd[each + 'len_aid_download_rate'].astype('float16')
			del train_pd[each]
		
		len_list = [x for x in train_pd.columns if 'len' in x]
		train_pd[len_list].to_hdf('../cache/str_len.hdf', 'w')
	print('len')


def get_label_feat_each_all(data, feat_set, cates, cat, col_num):
	each_col = [cat + '_' + str(i) for i in range(col_num)]
	fuck = -1
	for i, each in enumerate(each_col):
		if each == cates[-1]:
			fuck = i
	each_col[0], each_col[fuck] = each_col[fuck], each_col[0]
	
	feature_data = feat_set.loc[:, cates[:-1] + ['label'] + each_col]
	
	click_count = feature_data.groupby(cates[:-1] + [each_col[0]], as_index=False)[cates[0]].agg(
		{"_".join(cates) + "_click_count": 'count'})
	download_count = feature_data.loc[feature_data.label == 1].groupby(cates[:-1] + [each_col[0]], as_index=False)[
		cates[0]].agg(
		{"_".join(cates) + "_download_count": 'count'})
	
	for i in range(col_num):
		click_count_each = feature_data.groupby(cates[:-1] + [each_col[i]], as_index=False)[cates[0]].agg(
			{"_".join(cates) + "_click_count_": 'count'})
		click_count_each.rename(columns={each_col[i]: cates[-1]}, inplace=True)
		download_count_each = \
			feature_data.loc[feature_data.label == 1].groupby(cates[:-1] + [each_col[i]], as_index=False)[cates[0]].agg(
				{"_".join(cates) + "_download_count_": 'count'})
		download_count_each.rename(columns={each_col[i]: cates[-1]}, inplace=True)
		
		click_count = click_count.merge(click_count_each, 'left', cates).fillna(0)[cates +
		                                                                           ["_".join(
			                                                                           cates) + "_click_count"] + [
			                                                                           "_".join(
				                                                                           cates) + "_click_count_"]]
		
		download_count = download_count.merge(download_count_each, 'left', cates).fillna(0)[cates +
		                                                                                    ["_".join(
			                                                                                    cates) + "_download_count"] + [
			                                                                                    "_".join(
				                                                                                    cates) + "_download_count_"]]
		
		click_count["_".join(cates) + "_click_count"] = click_count["_".join(cates) + "_click_count"] + \
		                                                click_count["_".join(cates) + "_click_count_"]
		del click_count["_".join(cates) + "_click_count_"]
		download_count["_".join(cates) + "_download_count"] = download_count["_".join(cates) + "_download_count"] + \
		                                                      download_count["_".join(cates) + "_download_count_"]
		del download_count["_".join(cates) + "_download_count_"]
	
	count = click_count.merge(download_count, 'left', cates).fillna(0)
	
	count["_".join(cates) + "_click_rate"] = count["_".join(cates) + "_download_count"] / (
		count["_".join(cates) + "_click_count"] + 0.0001)
	data = data.merge(count[cates + ["_".join(cates) + "_click_rate", "_".join(cates) + "_download_count"]], 'left',
	                  cates).fillna(-1)
	gc.collect()
	return data["_".join(cates) + "_click_rate"], data["_".join(cates) + "_download_count"]


@cache
def get_label_feat_all(data, feat_set, cates, cate, col_num):
	folds = 5
	result = np.zeros((feat_set.shape[0], 1))
	result_1 = np.zeros((feat_set.shape[0], 1))
	kf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=520).split(feat_set, feat_set['label'])
	for i, (train_fold, test_fold) in enumerate(kf):
		result[test_fold, 0], result_1[test_fold, 0] = get_label_feat_each_all(feat_set.loc[test_fold, :].copy(),
		                                                                       feat_set.loc[train_fold, :].copy(),
		                                                                       cates, cate, col_num)
	label_fold, label_fold_1 = get_label_feat_each_all(data[data['label'] == -1].copy(), feat_set.copy(), cates, cate,
	                                                   col_num)
	label_fold_1 = label_fold_1 / 1.0 / folds * (folds - 1)
	data["_".join(cates) + "_click_rate_all"] = [x[0] for x in list(result)] + list(label_fold)
	data["_".join(cates) + "_click_count_all"] = [x[0] for x in list(result_1)] + list(label_fold_1)
	return data["_".join(cates) + "_click_rate_all"].astype('float16')


def safe_get(x, y):
	if y < len(x):
		return x[y]
	else:
		return '-1'


def str_label_feat_all_hdf(train_pd, caches):
	if os.path.exists('../cache/str_label_feat_all.hdf') and caches:
		print('load:  ../cache/str_label_feat_all.hdf')
	else:
		# str_feat = ['interest1', 'interest2', 'interest3', 'interest4', 'interest5', 'kw1', 'kw2', 'kw3', 'topic1',
		#             'topic2', 'topic3',]
		str_feat = ['kw2', 'topic1', 'topic2']
		str_len = [5, 5, 5, 5, 5, 5, 5]
		for i, each in enumerate(str_feat):
			print('str_label_feat_all: ', each)
			a = open('datas.txt', 'r')
			the_data = a.readline()
			str_feat_set = pd.read_csv('../input/'+the_data+'/userFeature.csv', usecols=[each, 'uid'])
			train_pd = train_pd.merge(str_feat_set, 'left', ['uid']).fillna('-1')
			del str_feat_set
			gc.collect()
			
			for k in range(str_len[i]):
				train_pd[each + '_' + str(k)] = train_pd[each].map(lambda x: safe_get(str(x).split(' '), k))
			for k in range(str_len[i]):
				train_pd[each + '_' + str(k) + '_label_feat_all'] = get_label_feat_all(train_pd, train_pd[
					train_pd['label'] != -1], [each + '_' + str(k)], each, str_len[i]).astype('float16')
				train_pd[each + '_' + str(k) + '_aid_label_feat_all'] = get_label_feat_all(train_pd, train_pd[
					train_pd['label'] != -1], ['aid', each + '_' + str(k)], each, str_len[i]).astype('float16')
			for k in range(str_len[i]):
				del train_pd[each + '_' + str(k)]
			del train_pd[each]
			len_list = [x for x in train_pd.columns if '_label_feat_all' in x]
			train_pd[len_list].to_hdf('../cache1/str_label_feat_all' + each + '.hdf', 'w')
			for each in len_list:
				del train_pd[each]
			gc.collect()
	print('str_label_feat_all')


def get_topn(data_list, lens):
	dicts = {}
	for each in data_list:
		if each in dicts:
			dicts[each] += 1
		else:
			dicts[each] = 1
	result = sorted(dicts.items(), key=lambda x: x[1], reverse=True)
	result = [x[0] for x in result] + ['-2'] * 300
	return result[:lens]


def topn_mean_mean(train_pd, how):
	if how == 1:
		str_feat = ['kw2']
		str_len = [50, 50, 50]
	if how == 2:
		str_feat = ['topic2']
		str_len = [50, 50, 50]
	if how == 3:
		str_feat = ['interest2']
		str_len = [50, 50, 50]
	for i, each in enumerate(str_feat):
		print('topn_mean_mean: ', each)
		a = open('datas.txt', 'r')
		the_data = a.readline()
		str_feat_set = pd.read_csv('../input/'+the_data+'/userFeature.csv', usecols=[each, 'uid'])
		train_pd = train_pd.merge(str_feat_set, 'left', ['uid']).fillna('-1')[['uid', 'aid', each]]
		del str_feat_set
		gc.collect()
		aid_hot_id = {}
		for each_line in train_pd[['aid', each]].values:
			if each_line[0] in aid_hot_id:
				aid_hot_id[each_line[0]].extend(each_line[1].split(' '))
			else:
				aid_hot_id[each_line[0]] = each_line[1].split(' ')
		for id, ii in enumerate(aid_hot_id):
			print(id)
			aid_hot_id[ii] = get_topn(aid_hot_id[ii], str_len[i])
		for iii in range(str_len[i]):
			print(iii)
			train_pd[each + '_top_' + str(iii)] = list(
				map(lambda x, y: 1 if aid_hot_id[y][iii] in x else 0, train_pd[each], train_pd['aid']))
			train_pd[each + '_top_' + str(iii)] = train_pd[each + '_top_' + str(iii)].astype('int8')
			
			temp = train_pd.groupby(['aid'], as_index=False)[each + '_top_' + str(iii)].agg(
				{each + '_top_' + str(iii) + '_mean': 'mean'})
			train_pd = train_pd.merge(temp, 'left', ['aid']).fillna(0)
			train_pd[each + '_top_' + str(iii) + '_mean'] = train_pd[each + '_top_' + str(iii) + '_mean'].astype(
				'float16')
			
			temp = train_pd.groupby(['uid'], as_index=False)[each + '_top_' + str(iii) + '_mean'].agg(
				{each + '_top_' + str(iii) + '_mean_means': 'mean'})
			train_pd = train_pd.merge(temp, 'left', ['uid']).fillna(0)
			train_pd[each + '_top_' + str(iii) + '_mean_means'] = train_pd[
				each + '_top_' + str(iii) + '_mean_means'].astype(
				'float16')
		len_list = [x for x in train_pd.columns if '_top_' in x and each in x]
		train_pd[len_list].to_hdf('../cache/topn_' + each + '.hdf', 'w')
		print('hao le')


def get_ids(x, i):
	a = x.split(' ')
	if i < len(a):
		return a[i]
	else:
		return '-1'


def split_str_feat(train_pd, how):
	str_feat = ['interest1', 'interest2', 'interest5', 'kw1', 'kw2', 'topic1', 'topic2']
	str_len = [10, 5, 10, 5, 5, 5, 5]
	for i, each in enumerate(str_feat):
		print(each)
		a = open('datas.txt', 'r')
		the_data = a.readline()
		str_feat_set = pd.read_csv('../input/'+the_data+'/userFeature.csv', usecols=[each, 'uid'])
		train_pd = train_pd.merge(str_feat_set, 'left', ['uid']).fillna('-1')
		del str_feat_set
		for iii in range(str_len[i]):
			print(iii)
			train_pd[each + '_hash_' + str(iii)] = train_pd[each].map(lambda x: get_ids(x, iii)).astype('int32')
		del train_pd[each]
	len_list = [x for x in train_pd.columns if '_hash_' in x]
	train_pd[len_list].to_hdf('../cache/str_hash.hdf', 'w')
	from sklearn.metrics import f1_score