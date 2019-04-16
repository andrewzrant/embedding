
# coding: utf-8

# In[1]:


# In[ ]:
import pandas as pd


# In[2]:


scores_train = pd.read_csv(
   'http://files.grouplens.org/datasets/movielens/ml-100k/ua.base', names=["uid", "mid", "rating", "timestamp"], sep="\t")
scores_test = pd.read_csv(
   'http://files.grouplens.org/datasets/movielens/ml-100k/ua.test', names=["uid", "mid", "rating", "timestamp"], sep="\t")


# In[3]:


import numpy as np

X_train = np.zeros((scores_train["mid"].max(), scores_train["uid"].max()))
for i, item in scores_train.iterrows():
    X_train[item["mid"] - 1, item["uid"] - 1] = item["rating"]


# In[4]:


X_train


# In[5]:


from sklearn.metrics.pairwise import pairwise_distances
X_cosine_train = 1-pairwise_distances(X_train, metric="cosine")
X_jaccard_train = 1-pairwise_distances(X_train, metric="jaccard")


# In[6]:


# 対角成分を0にする
np.fill_diagonal(X_cosine_train,0)
np.fill_diagonal(X_jaccard_train,0)


# In[7]:


X_cosine_train


# In[8]:


X_cosine_train.shape


# In[9]:


# ユーザの閲覧履歴を取る

user_item_dict_train = {}
user_item_rating_dict_train = {}
groups = scores_train.groupby('uid')
for i in range(1,943): # ユーザ数
    user_item_dict_train[i] = groups.get_group(i)['mid'].values
    user_item_rating_dict_train[i] = groups.get_group(i)['rating'].values


# In[10]:


# ユーザの閲覧履歴(test)

user_item_dict_test = {}
user_item_rating_dict_test = {}
groups = scores_test.groupby('uid')
for i in range(1,943): # ユーザ数
    user_item_dict_test[i] = groups.get_group(i)['mid'].values
    user_item_rating_dict_test[i] = groups.get_group(i)['rating'].values


# In[11]:


np.zeros(1682)


# In[12]:


# ユーザの閲覧履歴に合わせてsimilarityを足し合わせる(rating3以上)
from tqdm import tqdm

user_item_sim_dict_tanimoto = {}
user_item_sim_dict_cosine = {}
for i in tqdm(range(1,943)):
    user_item_sim_dict_tanimoto[i] = np.zeros(X_jaccard_train.shape[1])
    user_item_sim_dict_cosine[i] = np.zeros(X_cosine_train.shape[1])
    item_history = user_item_dict_train[i]
    rating_history = user_item_rating_dict_train[i]
    for item,rating in zip(item_history,rating_history):
        if rating >= 3:
            user_item_sim_dict_tanimoto[i] += X_jaccard_train[item-1]
            user_item_sim_dict_cosine[i] += X_cosine_train[item-1]


# In[18]:


# ユーザごとにTOP10を出してRecall@10を計算する
def precision_and_recall(ranked_list,ground_list):
    hits = 0
    for i in range(len(ranked_list)):
        id = ranked_list[i]
        if id in ground_list:
            hits += 1
    pre = hits/(1.0 * len(ranked_list))
    rec = hits/(1.0 * len(ground_list))
    return pre, rec


# ## Tanimoto(Jaccard)
# @10

# In[19]:


precision_list = []
recall_list = []

for i in range(1,943):
    ranked_list = [v+1 for v in np.argsort(user_item_sim_dict_tanimoto[i])[::-1][:10]] # midは1から始まるため
    ground_rating = user_item_rating_dict_test[i]
    ground_list = user_item_dict_test[i]
    ground_list_cut = [w for v,w in zip(ground_rating,ground_list) if v >= 3]
    
    pre, rec = precision_and_recall(ranked_list,ground_list_cut)
    precision_list.append(pre)
    recall_list.append(rec)


# In[21]:


precision = sum(precision_list) / len(precision_list)
recall = sum(recall_list) / len(recall_list)
f1 = 2 * precision * recall / (precision + recall)


# In[22]:


print(f'precision: {precision}')
print(f'recall: {recall}')
print(f'f1: {f1}')


# ## Cosine
# @10

# In[23]:


precision_list = []
recall_list = []

for i in range(1,943):
    ranked_list = [v+1 for v in np.argsort(user_item_sim_dict_cosine[i])[::-1][:10]] # midは1から始まるため
    ground_list = user_item_dict_test[i]
    ground_rating = user_item_rating_dict_test[i]
    ground_list_cut = [w for v,w in zip(ground_rating,ground_list) if v >= 3]
    
    pre, rec = precision_and_recall(ranked_list,ground_list)
    precision_list.append(pre)
    recall_list.append(rec)


# In[24]:


precision = sum(precision_list) / len(precision_list)
recall = sum(recall_list) / len(recall_list)
f1 = 2 * precision * recall / (precision + recall)


# In[25]:


print(f'precision: {precision}')
print(f'recall: {recall}')
print(f'f1: {f1}')


# # Item2Vec

# In[26]:


train_values = []
for key in range(1,943):
    tmp =  [str(v) for v,w in zip(user_item_dict_train[key],user_item_rating_dict_train[key]) if w >= 3]
    train_values.append(tmp)


# In[27]:


import logging
import os.path
import sys
 
logger = logging.getLogger()
  
logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s")
logging.root.setLevel(level=logging.INFO)


# In[28]:


from gensim.models import word2vec
# word2vec
#     sg: 1=skip-gram
#     window: ウィンドウサイズ、全アイテムをコンテキストに含めたいのでとびきり大きな値
#     hs: 0=negative sampling(negativeはdefault)
model = word2vec.Word2Vec(train_values, sg=1, size=128, window=100000, hs=0,  seed=0)


# In[29]:


user_vectors = {}

for i in range(1,943):
    tmp_vector = np.empty((2,128))
    item_history = user_item_dict_train[i]
    rating_history = user_item_rating_dict_train[i]
    for item,rating in zip(item_history,rating_history):
        try:
            tmp_vector += model.wv[str(item)] * (rating/3)
        except:
            continue
        
    tmp_vector /= len(item_history)
    user_vectors[i] = tmp_vector


# In[30]:


precision_list = []
recall_list = []

for i in range(1,943):
    ranked_list = [int(v[0]) for v in model.most_similar(user_vectors[i], [], 10)]
    ground_list = user_item_dict_test[i]
    ground_rating = user_item_rating_dict_test[i]
    ground_list_cut = [w for v,w in zip(ground_rating,ground_list) if v >= 3]
    pre, rec = precision_and_recall(ranked_list,ground_list_cut)
    precision_list.append(pre)
    recall_list.append(rec)


# In[32]:


precision = sum(precision_list) / len(precision_list)
recall = sum(recall_list) / len(recall_list)
f1 = 2 * precision * recall / (precision + recall)


# In[33]:


print(f'precision: {precision}')
print(f'recall: {recall}')
print(f'f1: {f1}')


# ## xdeepdm
# @10

# In[52]:


import tensorflow as tf
import ctrNet
from src import misc_utils as utils
from scipy.sparse import lil_matrix


# In[46]:


nbUsers = len(set(pd.concat([scores_train['uid'],scores_test['uid']])))
nbMovies=len(set(pd.concat([scores_train['mid'],scores_test['mid']])))
nbFeatures=nbUsers+nbMovies
nbRatingsTrain=len(scores_train)
nbRatingsTest=len(scores_test)


# In[78]:


def loadDataset(df, nbUsers, lines, columns):
    X = np.zeros((lines, columns))
    Y = []
    for index, row in df.iterrows():
        userId = row['uid']
        movieId = row['mid']
        rating = row['rating']
        X[index,int(userId)-1] = 1
        X[index,int(nbUsers)+int(movieId)-1] = 1
        if int(rating) >= 3:
            Y.append(1)
        else:
            Y.append(0)
            
    Y=np.array(Y).astype('float32')
    X = pd.DataFrame(X)
    X.columns=['f'+str(i) for i in range(0, columns)]
    return X, Y


# In[96]:


X_train_xdeepfm, y_train_xdeepfm = loadDataset(scores_train, nbUsers,nbRatingsTrain, nbFeatures)
X_test_xdeepfm, y_test_xdeepfm = loadDataset(scores_test, nbUsers, nbRatingsTrain, nbFeatures)


# In[97]:


from sklearn.model_selection import train_test_split
X_train_xdeepfm, X_valid_xdeepfm, y_train_xdeepfm, y_valid_xdeepfm = train_test_split(X_train_xdeepfm, y_train_xdeepfm, test_size=0.3, random_state=42)


# In[84]:


hparam = tf.contrib.training.HParams(
    model='xdeepfm',
    norm=True,
    batch_norm_decay=0.9,
    hidden_size=[128, 128],
    cross_layer_sizes=[128, 128, 128],
    k=8,
    hash_ids=int(2e5),
    batch_size=64,
    optimizer="adam",
    learning_rate=0.001,
    num_display_steps=1000,
    num_eval_steps=1000,
    epoch=1,
    metric='auc',
    activation=['relu', 'relu', 'relu'],
    cross_activation='identity',
    init_method='uniform',
    init_value=0.1,
    feature_nums=len(features))
utils.print_hparams(hparam)


# In[85]:


os.environ["CUDA_DEVICE_ORDER"]='PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"]='0'
model=ctrNet.build_model(hparam)


# In[ ]:


model.train(train_data=(X_train_xdeepfm, y_train_xdeepfm),            dev_data=(X_valid_xdeepfm, y_valid_xdeepfm))

