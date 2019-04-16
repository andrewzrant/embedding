
# coding: utf-8

# In[60]:


# In[ ]:
import pandas as pd
import numpy as np


# In[61]:


COL_NAME = ['uid','mid','rating','timestamp']
scores = pd.read_csv("data/ml-1m/ratings.dat", sep="::", header=None, engine='python', names=COL_NAME)


# In[62]:


from sklearn.model_selection import train_test_split
scores_train, scores_test = train_test_split(scores, test_size=0.2, random_state=42)


# In[4]:


X_train = np.zeros((scores_train["mid"].max(), scores_train["uid"].max()))
for i, item in scores_train.iterrows():
    X_train[item["mid"] - 1, item["uid"] - 1] = item["rating"]


# In[5]:


X_train


# In[6]:


from sklearn.metrics.pairwise import pairwise_distances
X_cosine_train = 1-pairwise_distances(X_train, metric="cosine")
X_jaccard_train = 1-pairwise_distances(X_train, metric="jaccard")


# In[7]:


# 対角成分を0にする
np.fill_diagonal(X_cosine_train,0)
np.fill_diagonal(X_jaccard_train,0)


# In[8]:


X_cosine_train


# In[9]:


# ユーザの閲覧履歴を取る

user_item_dict_train = {}
user_item_rating_dict_train = {}
groups = scores_train.groupby('uid')
user_item_dict_train = groups["mid"].apply(lambda x: x.tolist()).to_dict()
user_item_rating_dict_train = groups["rating"].apply(lambda x: x.tolist()).to_dict()


# In[10]:


# ユーザの閲覧履歴(test)

user_item_dict_test = {}
user_item_rating_dict_test = {}
groups = scores_test.groupby('uid')
user_item_dict_test = groups["mid"].apply(lambda x: x.tolist()).to_dict()
user_item_rating_dict_test = groups["rating"].apply(lambda x: x.tolist()).to_dict()


# In[11]:


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


# In[12]:


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

# In[78]:


precision_list = []
recall_list = []

for i in user_item_sim_dict_tanimoto.keys():
    if i in user_item_dict_test.keys():
        ranked_list = [v+1 for v in np.argsort(user_item_sim_dict_tanimoto[i])[::-1][:10]] # midは1から始まるため
        ground_rating = user_item_rating_dict_test[i]
        ground_list = user_item_dict_test[i]
        ground_list_cut = [w for v,w in zip(ground_rating,ground_list) if v >= 3]
        
        if len(ground_list_cut) > 0:
            pre, rec = precision_and_recall(ranked_list,ground_list_cut)
            precision_list.append(pre)
            recall_list.append(rec)


# In[79]:


precision = sum(precision_list) / len(precision_list)
recall = sum(recall_list) / len(recall_list)
f1 = 2 * precision * recall / (precision + recall)


# In[80]:


print(f'precision: {precision}')
print(f'recall: {recall}')
print(f'f1: {f1}')


# ## Cosine
# @10

# In[81]:


precision_list = []
recall_list = []

for i in user_item_sim_dict_cosine.keys():
    if i in user_item_dict_test.keys():
        ranked_list = [v+1 for v in np.argsort(user_item_sim_dict_cosine[i])[::-1][:10]] # midは1から始まるため
        ground_rating = user_item_rating_dict_test[i]
        ground_list = user_item_dict_test[i]
        ground_list_cut = [w for v,w in zip(ground_rating,ground_list) if v >= 3]
        
        if len(ground_list_cut) > 0:
            pre, rec = precision_and_recall(ranked_list,ground_list_cut)
            precision_list.append(pre)
            recall_list.append(rec)


# In[82]:


precision = sum(precision_list) / len(precision_list)
recall = sum(recall_list) / len(recall_list)
f1 = 2 * precision * recall / (precision + recall)


# In[83]:


print(f'precision: {precision}')
print(f'recall: {recall}')
print(f'f1: {f1}')


# # Item2Vec

# In[19]:


import logging
import os.path
import sys
 
logger = logging.getLogger()
  
logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s")
logging.root.setLevel(level=logging.INFO)


# In[38]:


def functor(f, l):
    if isinstance(l,list):
        return [functor(f,i) for i in l]
    else:
        return f(l)


# In[40]:


from gensim.models import word2vec
# word2vec
#     sg: 1=skip-gram
#     window: ウィンドウサイズ、全アイテムをコンテキストに含めたいのでとびきり大きな値
#     hs: 0=negative sampling(negativeはdefault)
model = word2vec.Word2Vec(functor(str, list(user_item_dict_test.values())), sg=1, size=128, window=100000, hs=0,  seed=0)


# In[41]:


user_vectors = {}

for i in user_item_dict_train.keys():
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


# In[43]:


[int(v[0]) for v in model.most_similar(user_vectors[i], [], 10)]


# In[84]:


precision_list = []
recall_list = []

for i in user_item_dict_train.keys():
    if i in user_item_dict_test.keys():
        ranked_list = [int(v[0]) for v in model.most_similar(user_vectors[i], [], 10)]
        ground_list = user_item_dict_test[i]
        ground_rating = user_item_rating_dict_test[i]
        ground_list_cut = [w for v,w in zip(ground_rating,ground_list) if v >= 3]
        
        if len(ground_list_cut) > 0:
            pre, rec = precision_and_recall(ranked_list, ground_list_cut)
            precision_list.append(pre)
            recall_list.append(rec)


# In[85]:


precision = sum(precision_list) / len(precision_list)
recall = sum(recall_list) / len(recall_list)
f1 = 2 * precision * recall / (precision + recall)


# In[86]:


print(f'precision: {precision}')
print(f'recall: {recall}')
print(f'f1: {f1}')


# ## deepfm
# @10

# In[63]:


import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def load_movies():
    COL_NAME = ['mid','movie_name','movie_genre']
    df = pd.read_csv('data/ml-1m/movies.dat',sep='::', header=None, engine='python', names=COL_NAME)
    return df

def load_users():
    COL_NAME = ['uid','user_fea1','user_fea2','user_fea3','user_fea4']
    df = pd.read_csv('data/ml-1m/users.dat',sep='::', header=None, engine='python', names=COL_NAME)
    return df

def text2seq(text, n_genre):
    """ using tokenizer to encoded the multi-level categorical feature
    """
    tokenizer = Tokenizer(lower=True, split='|',filters='', num_words=n_genre)
    tokenizer.fit_on_texts(text)
    seq = tokenizer.texts_to_sequences(text)
    seq = pad_sequences(seq, maxlen=3,padding='post')
    return seq

n_genre = 15

movies = load_movies()
users = load_users()

print("===== movies.dat ======")
print(movies.head())
print("====== users.dat ======")
print(users.head())

movies['movie_genre'] = text2seq(movies.movie_genre.values, n_genre=n_genre).tolist()

scores = scores.join(movies.set_index('mid'), on = 'mid', how = 'left')
scores = scores.join(users.set_index('uid'), on = 'uid', how = 'left')

scores_train = scores_train.join(movies.set_index('mid'), on = 'mid', how = 'left')
scores_train = scores_train.join(users.set_index('uid'), on = 'uid', how = 'left')


# In[131]:


movies.head(100)


# In[132]:


# テスト用にuser, item全組み合わせを作る
all_user = []
all_item = []
for u in list(users["uid"]):
    for m in list(movies["mid"]):
        all_user.append(u)
        all_item.append(m)
fm_test = pd.DataFrame([])
fm_test["uid"] = all_user
fm_test["mid"] = all_item
fm_test = fm_test.join(movies.set_index('mid'), on = 'mid', how = 'left')
fm_test = fm_test.join(users.set_index('uid'), on = 'uid', how = 'left')


# In[133]:


fm_test


# In[65]:


import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *

def define_input_layers():
    # numerica features
    fea3_input = Input((1,), name = 'input_fea3')
    num_inputs = [fea3_input]
    # single level categorical features
    uid_input = Input((1,), name = 'input_uid')
    mid_input = Input((1,), name= 'input_mid')
    cat_sl_inputs = [uid_input, mid_input]

    # multi level categorical features (with 3 genres at most)
    genre_input = Input((3,), name = 'input_genre')
    cat_ml_inputs = [genre_input]

    inputs = num_inputs + cat_sl_inputs + cat_ml_inputs
    
    return inputs

inputs = define_input_layers()


# In[66]:


def Tensor_Mean_Pooling(name = 'mean_pooling', keepdims = False):
    return Lambda(lambda x: K.mean(x, axis = 1, keepdims=keepdims), name = name)

def fm_1d(inputs, n_uid, n_mid, n_genre):
    
    fea3_input, uid_input, mid_input, genre_input = inputs
    
    # all tensors are reshape to (None, 1)
    num_dense_1d = [Dense(1, name = 'num_dense_1d_fea4')(fea3_input)]
    cat_sl_embed_1d = [Embedding(n_uid + 1, 1, name = 'cat_embed_1d_uid')(uid_input),
                        Embedding(n_mid + 1, 1, name = 'cat_embed_1d_mid')(mid_input)]
    cat_ml_embed_1d = [Embedding(n_genre + 1, 1, mask_zero=True, name = 'cat_embed_1d_genre')(genre_input)]

    cat_sl_embed_1d = [Reshape((1,))(i) for i in cat_sl_embed_1d]
    cat_ml_embed_1d = [Tensor_Mean_Pooling(name = 'embed_1d_mean')(i) for i in cat_ml_embed_1d]
    
    # add all tensors
    y_fm_1d = Add(name = 'fm_1d_output')(num_dense_1d + cat_sl_embed_1d + cat_ml_embed_1d)
    
    return y_fm_1d

y_1d = fm_1d(inputs, 10, 10, 10)


# In[67]:


def fm_2d(inputs, n_uid, n_mid, n_genre, k):
    
    fea3_input, uid_input, mid_input, genre_input = inputs
    
    num_dense_2d = [Dense(k, name = 'num_dense_2d_fea3')(fea3_input)] # shape (None, k)
    num_dense_2d = [Reshape((1,k))(i) for i in num_dense_2d] # shape (None, 1, k)

    cat_sl_embed_2d = [Embedding(n_uid + 1, k, name = 'cat_embed_2d_uid')(uid_input), 
                       Embedding(n_mid + 1, k, name = 'cat_embed_2d_mid')(mid_input)] # shape (None, 1, k)
    
    cat_ml_embed_2d = [Embedding(n_genre + 1, k, name = 'cat_embed_2d_genre')(genre_input)] # shape (None, 3, k)
    cat_ml_embed_2d = [Tensor_Mean_Pooling(name = 'cat_embed_2d_genure_mean', keepdims=True)(i) for i in cat_ml_embed_2d] # shape (None, 1, k)

    # concatenate all 2d embed layers => (None, ?, k)
    embed_2d = Concatenate(axis=1, name = 'concat_embed_2d')(num_dense_2d + cat_sl_embed_2d + cat_ml_embed_2d)

    # calcuate the interactions by simplication
    # sum of (x1*x2) = sum of (0.5*[(xi)^2 - (xi^2)])
    tensor_sum = Lambda(lambda x: K.sum(x, axis = 1), name = 'sum_of_tensors')
    tensor_square = Lambda(lambda x: K.square(x), name = 'square_of_tensors')

    sum_of_embed = tensor_sum(embed_2d)
    square_of_embed = tensor_square(embed_2d)

    square_of_sum = Multiply()([sum_of_embed, sum_of_embed])
    sum_of_square = tensor_sum(square_of_embed)

    sub = Subtract()([square_of_sum, sum_of_square])
    sub = Lambda(lambda x: x*0.5)(sub)
    y_fm_2d = Reshape((1,), name = 'fm_2d_output')(tensor_sum(sub))
    
    return y_fm_2d, embed_2d

y_fm2_d, embed_2d = fm_2d(inputs, 10, 10, 10, 5)


# In[68]:


def deep_part(embed_2d, dnn_dim, dnn_dr):
    
    # flat embed layers from 3D to 2D tensors
    y_dnn = Flatten(name = 'flat_embed_2d')(embed_2d)
    for h in dnn_dim:
        y_dnn = Dropout(dnn_dr)(y_dnn)
        y_dnn = Dense(h, activation='relu')(y_dnn)
    y_dnn = Dense(1, activation='relu', name = 'deep_output')(y_dnn)
    
    return y_dnn

y_dnn = deep_part(embed_2d, [16, 16], 0.5)


# In[69]:


def deep_fm_model(n_uid, n_mid, n_genre, k, dnn_dim, dnn_dr):
    
    inputs = define_input_layers()
    
    y_fm_1d = fm_1d(inputs, n_uid, n_mid, n_genre)
    y_fm_2d, embed_2d = fm_2d(inputs, n_uid, n_mid, n_genre, k)
    y_dnn = deep_part(embed_2d, dnn_dim, dnn_dr)
    
    # combinded deep and fm parts
    y = Concatenate()([y_fm_1d, y_fm_2d, y_dnn])
    y = Dense(1, name = 'deepfm_output')(y)
    
    fm_model_1d = Model(inputs, y_fm_1d)
    fm_model_2d = Model(inputs, y_fm_2d)
    deep_model = Model(inputs, y_dnn)
    deep_fm_model = Model(inputs, y)
    
    return fm_model_1d, fm_model_2d, deep_model, deep_fm_model


# In[70]:


params = {
    'n_uid': scores.uid.max(),
    'n_mid': scores.mid.max(),
    'n_genre': 14,
    'k':20,
    'dnn_dim':[64,64],
    'dnn_dr': 0.5
}

fm_model_1d, fm_model_2d, deep_model, deep_fm_model = deep_fm_model(**params)


# In[173]:


def df2xy(ratings, label=True):
    x = [ratings.user_fea3.values, 
         ratings.uid.values, 
         ratings.mid.values, 
         np.concatenate(ratings.movie_genre.values).reshape(-1,3)]
    if label:
        y = ratings.rating.values
        return x,y
    else:
        return x

train_x, train_y = df2xy(scores_train)
test_x, test_y = df2xy(scores_test)


# In[134]:


fm_test_x = df2xy(fm_test, False)


# In[215]:


from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
# train  model
deep_fm_model.compile(loss = 'MSE', optimizer='adam')
early_stop = EarlyStopping(monitor='val_loss', patience=30, verbose=1)
model_ckp = ModelCheckpoint(filepath='deepfm_weights.h5', 
                            monitor='val_loss',
                            save_weights_only=True, 
                            save_best_only=True)
callbacks = [model_ckp,early_stop]
train_history = deep_fm_model.fit(train_x, train_y, 
                                  epochs=50, batch_size=2048,
                                  validation_split=0.1, 
                                  callbacks = callbacks)


# In[225]:


# RMSE計算
predict_y_score = deep_fm_model.predict(test_x, batch_size=2048)


# In[226]:


from sklearn.metrics import mean_squared_error
np.sqrt(mean_squared_error(test_y, predict_y_score))


# In[227]:


# Top-k Recommendation
# fmを用いてユーザーごとにratingの高いアイテムを表出する
fm_test_y = deep_fm_model.predict(fm_test_x, batch_size=2048)


# In[201]:


fm_test["prediction"] = [v[0] for v in fm_test_y]


# In[204]:


fm_test.to_csv("output/fm_test.csv")


# In[205]:


# uidごとにmidをpredictionに沿ってsort
fm_test = fm_test.sort_values(['uid', 'prediction'], ascending=[True, False])


# In[206]:


fm_test[(fm_test['uid']==3)&(fm_test['mid']==2081)]


# In[218]:


# 辞書としてtop10のmidをuidに紐付ける
groups = fm_test.groupby('uid')
user_item_dict_predict = groups["mid"].apply(lambda x: x.tolist()[:100]).to_dict()


# In[223]:


set(user_item_dict_train[3])&set(user_item_dict_predict[3])


# In[224]:


user_item_dict_predict[3]


# In[219]:


precision_list = []
recall_list = []

for i in user_item_dict_train.keys():
    if i in user_item_dict_test.keys():
        ranked_list = user_item_dict_predict[i]
        ground_list = user_item_dict_test[i]
        ground_rating = user_item_rating_dict_test[i]
        ground_list_cut = [w for v,w in zip(ground_rating,ground_list) if v >= 3]
        
        if len(ground_list_cut) > 0:
            pre, rec = precision_and_recall(ranked_list, ground_list_cut)
            precision_list.append(pre)
            recall_list.append(rec)


# In[220]:


precision = sum(precision_list) / len(precision_list)
recall = sum(recall_list) / len(recall_list)
f1 = 2 * precision * recall / (precision + recall)


# In[221]:


print(f'precision: {precision}')
print(f'recall: {recall}')
print(f'f1: {f1}')


# In[160]:


weights = deep_fm_model.get_weights()
fm_1_weight, fm_2d_weigth, deep_weight = weights[-2]
print("""
contribution of different part of model
    weight of 1st order fm: %5.3f
    weight of 2nd order fm: %5.3f
    weight of dnn part: %5.3f
""" % (fm_1_weight, fm_2d_weigth, deep_weight))


# ## LightFM
# なんとかFMでめっちゃ精度出したい
# たぶん、deepFMは自分の実装が間違っていただけ...と信じる

# In[228]:


scores_train, scores_test = train_test_split(scores, test_size=0.2, random_state=42)


# In[229]:


from scipy.sparse import lil_matrix


# In[257]:


COL_NAME = ['mid','movie_name','movie_genre']
movies = pd.read_csv('data/ml-1m/movies.dat',sep='::', header=None, engine='python', names=COL_NAME)


# In[258]:


movies


# In[239]:


rows = users["uid"].max() + 1
cols = movies["mid"].max() + 1


# In[241]:


mat_train = lil_matrix((rows, cols), dtype=np.int32)

for index, row in scores_train.iterrows():
    uid = row['uid']
    mid = row['mid']
    rating = row['rating']
    if rating >= 3.0:
        mat_train[uid, mid] = 1.0
    else:
        mat_train[uid, mid] = -1.0


# In[242]:


mat_test = lil_matrix((rows, cols), dtype=np.int32)

for index, row in scores_test.iterrows():
    uid = row['uid']
    mid = row['mid']
    rating = row['rating']
    if rating >= 3.0:
        mat_test[uid, mid] = 1.0
    else:
        mat_test[uid, mid] = -1.0


# In[272]:


genre_set = list(set(sum(movies['movie_genre'].apply(lambda x: x.split("|")).values, [])))
genre_set_dict = {}
for i in range(len(genre_set)):
    genre_set_dict[genre_set[i]] = i


# In[275]:


features = {}
for index, row in movies.iterrows():
    item_id = int(row['mid'])
    
    genres = row['movie_genre'].split("|")
    genre_ids = [genre_set_dict[v]for v in genres]

#     if use_item_ids:
#         # Add item-specific features too
#         genres.append(item_id)

    features[item_id] = genre_ids


# In[280]:


mat_items = lil_matrix((movies["mid"].max() + 1, len(genre_set)), dtype=np.int32)

for item_id, genre_ids in features.items():
    for genre_id in genre_ids:
        mat_items[item_id, genre_id] = 1


# In[247]:


from lightfm import LightFM
from lightfm.evaluation import precision_at_k, recall_at_k
from lightfm.evaluation import auc_score


# In[281]:


model = LightFM(learning_rate=0.05, loss='warp', learning_schedule='adagrad')
model.fit(mat_train, item_features=mat_items, epochs=100)


# In[ ]:


train_precision = precision_at_k(model, mat_train, k=10, item_features=mat_items).mean()
test_precision = precision_at_k(model, mat_test, k=10, item_features=mat_items).mean()

train_recall = recall_at_k(model, mat_train, k=10, item_features=mat_items).mean()
test_recall = recall_at_k(model, mat_test, k=10, item_features=mat_items).mean()

train_auc = auc_score(model, mat_train, item_features=mat_items).mean()
test_auc = auc_score(model, mat_test, item_features=mat_items).mean()

print('Precision: train %.2f, test %.2f.' % (train_precision, test_precision))
print('Recall: train %.2f, test %.2f.' % (train_recall, test_recall))
print('F1: train %.2f, test %.2f.' % ((2*train_recall*train_precision)/(train_recall+train_precision), (2*test_recall*test_precision)/(test_recall+test_precision)))
print('AUC: train %.2f, test %.2f.' % (train_auc, test_auc))

