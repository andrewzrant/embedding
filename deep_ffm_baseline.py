import os
import time
import pandas as pd
os.system('pip install tensorflow-gpu')


#os.system('pip install xgboost -gpu')
os.system('pip install pytime')

import numpy as np
import random
seed = random.randint(2000, 3000)
print(seed)

class Unbuffered(object):
   def __init__(self, stream):
       self.stream = stream
   def write(self, data):
       self.stream.write(data)
       self.stream.flush()
   def __getattr__(self, attr):
       return getattr(self.stream, attr)
import sys
sys.stdout = Unbuffered(sys.stdout)
os.system('free -h')
os.system('cat /proc/cpuinfo| grep "processor"| wc -l')


import warnings
import gc
import pandas as pd
from time import time
from math import sqrt
import scipy.stats as st
import os
from pytime import pytime
import datetime
from datetime import timedelta
from scipy.special import boxcox1p
from scipy import sparse
import numpy as np
from sklearn.preprocessing import MinMaxScaler
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore")

os.chdir("/cos_person/Elo/data" )
import numpy as np
import tensorflow as tf
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import roc_auc_score
from time import time
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm



class DeepFM(BaseEstimator, TransformerMixin):
    def __init__(self, feature_size, field_size,
                 embedding_size=8, dropout_fm=[1.0, 1.0],
                 deep_layers=[32, 32], dropout_deep=[0.5, 0.5, 0.5],
                 deep_layers_activation=tf.nn.relu,
                 epoch=10, batch_size=256,
                 learning_rate=0.001, optimizer_type="adam",
                 batch_norm=0, batch_norm_decay=0.995,
                 verbose=False, random_seed=2016,
                 use_fm=True, use_deep=True,
                 loss_type="logloss", eval_metric=roc_auc_score,
                 l2_reg=0.0, greater_is_better=True):
        assert (use_fm or use_deep)
        assert loss_type in ["logloss", "mse"], \
            "loss_type can be either 'logloss' for classification task or 'mse' for regression task"

        self.feature_size = feature_size        # denote as M, size of the feature dictionary
        self.field_size = field_size            # denote as F, size of the feature fields
        self.embedding_size = embedding_size    # denote as K, size of the feature embedding

        self.dropout_fm = dropout_fm
        self.deep_layers = deep_layers
        self.dropout_deep = dropout_deep
        self.deep_layers_activation = deep_layers_activation
        self.use_fm = use_fm
        self.use_deep = use_deep
        self.l2_reg = l2_reg

        self.epoch = epoch
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.optimizer_type = optimizer_type

        self.batch_norm = batch_norm
        self.batch_norm_decay = batch_norm_decay

        self.verbose = verbose
        self.random_seed = random_seed
        self.loss_type = loss_type
        self.eval_metric = eval_metric
        self.greater_is_better = greater_is_better
        self.train_result, self.valid_result = [], []

        self._init_graph()


    def _init_graph(self):
        self.graph = tf.Graph()
        with self.graph.as_default():

            tf.set_random_seed(self.random_seed)

            self.feat_index = tf.placeholder(tf.int32, shape=[None, None],
                                                 name="feat_index")  # None * F
            self.feat_value = tf.placeholder(tf.float32, shape=[None, None],
                                                 name="feat_value")  # None * F
            self.label = tf.placeholder(tf.float32, shape=[None, 1], name="label")  # None * 1
            self.dropout_keep_fm = tf.placeholder(tf.float32, shape=[None], name="dropout_keep_fm")
            self.dropout_keep_deep = tf.placeholder(tf.float32, shape=[None], name="dropout_keep_deep")
            self.train_phase = tf.placeholder(tf.bool, name="train_phase")

            self.weights = self._initialize_weights()

            # model
            self.embeddings = tf.nn.embedding_lookup(self.weights["feature_embeddings"],
                                                             self.feat_index)  # None * F * K
            feat_value = tf.reshape(self.feat_value, shape=[-1, self.field_size, 1])
            self.embeddings = tf.multiply(self.embeddings, feat_value)

            # ---------- first order term ----------
            self.y_first_order = tf.nn.embedding_lookup(self.weights["feature_bias"], self.feat_index) # None * F * 1
            self.y_first_order = tf.reduce_sum(tf.multiply(self.y_first_order, feat_value), 2)  # None * F
            self.y_first_order = tf.nn.dropout(self.y_first_order, self.dropout_keep_fm[0]) # None * F

            # ---------- second order term ---------------
            # sum_square part
            self.summed_features_emb = tf.reduce_sum(self.embeddings, 1)  # None * K
            self.summed_features_emb_square = tf.square(self.summed_features_emb)  # None * K

            # square_sum part
            self.squared_features_emb = tf.square(self.embeddings)
            self.squared_sum_features_emb = tf.reduce_sum(self.squared_features_emb, 1)  # None * K

            # second order
            self.y_second_order = 0.5 * tf.subtract(self.summed_features_emb_square, self.squared_sum_features_emb)  # None * K
            self.y_second_order = tf.nn.dropout(self.y_second_order, self.dropout_keep_fm[1])  # None * K

            # ---------- Deep component ----------
            self.y_deep = tf.reshape(self.embeddings, shape=[-1, self.field_size * self.embedding_size]) # None * (F*K)
            self.y_deep = tf.nn.dropout(self.y_deep, self.dropout_keep_deep[0])
            for i in range(0, len(self.deep_layers)):
                self.y_deep = tf.add(tf.matmul(self.y_deep, self.weights["layer_%d" %i]), self.weights["bias_%d"%i]) # None * layer[i] * 1
                if self.batch_norm:
                    self.y_deep = self.batch_norm_layer(self.y_deep, train_phase=self.train_phase, scope_bn="bn_%d" %i) # None * layer[i] * 1
                self.y_deep = self.deep_layers_activation(self.y_deep)
                self.y_deep = tf.nn.dropout(self.y_deep, self.dropout_keep_deep[1+i]) # dropout at each Deep layer

            # ---------- DeepFM ----------
            if self.use_fm and self.use_deep:
                concat_input = tf.concat([self.y_first_order, self.y_second_order, self.y_deep], axis=1)
            elif self.use_fm:
                concat_input = tf.concat([self.y_first_order, self.y_second_order], axis=1)
            elif self.use_deep:
                concat_input = self.y_deep
            self.out = tf.add(tf.matmul(concat_input, self.weights["concat_projection"]), self.weights["concat_bias"])

            # loss
            if self.loss_type == "logloss":
                self.out = tf.nn.sigmoid(self.out)
                self.loss = tf.losses.log_loss(self.label, self.out)
            elif self.loss_type == "mse":
                self.loss = tf.nn.l2_loss(tf.subtract(self.label, self.out))
            # l2 regularization on weights
            if self.l2_reg > 0:
                self.loss += tf.contrib.layers.l2_regularizer(
                    self.l2_reg)(self.weights["concat_projection"])
                if self.use_deep:
                    for i in range(len(self.deep_layers)):
                        self.loss += tf.contrib.layers.l2_regularizer(
                            self.l2_reg)(self.weights["layer_%d"%i])

            # optimizer
            if self.optimizer_type == "adam":
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.999,
                                                        epsilon=1e-8).minimize(self.loss)
            elif self.optimizer_type == "adagrad":
                self.optimizer = tf.train.AdagradOptimizer(learning_rate=self.learning_rate,
                                                           initial_accumulator_value=1e-8).minimize(self.loss)
            elif self.optimizer_type == "gd":
                self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
            elif self.optimizer_type == "momentum":
                self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=0.95).minimize(
                    self.loss)
            elif self.optimizer_type == "yellowfin":
                self.optimizer = YFOptimizer(learning_rate=self.learning_rate, momentum=0.0).minimize(
                    self.loss)

            # init
            self.saver = tf.train.Saver()
            init = tf.global_variables_initializer()
            self.sess = self._init_session()
            self.sess.run(init)

            # number of params
            total_parameters = 0
            for variable in self.weights.values():
                shape = variable.get_shape()
                variable_parameters = 1
                for dim in shape:
                    variable_parameters *= dim.value
                total_parameters += variable_parameters
            if self.verbose > 0:
                print("#params: %d" % total_parameters)


    def _init_session(self):
        config = tf.ConfigProto(device_count={"gpu": 0})
        config.gpu_options.allow_growth = True
        return tf.Session(config=config)


    def _initialize_weights(self):
        weights = dict()

        # embeddings
        weights["feature_embeddings"] = tf.Variable(
            tf.random_normal([self.feature_size, self.embedding_size], 0.0, 0.01),
            name="feature_embeddings")  # feature_size * K
        weights["feature_bias"] = tf.Variable(
            tf.random_uniform([self.feature_size, 1], 0.0, 1.0), name="feature_bias")  # feature_size * 1

        # deep layers
        num_layer = len(self.deep_layers)
        input_size = self.field_size * self.embedding_size
        glorot = np.sqrt(2.0 / (input_size + self.deep_layers[0]))
        weights["layer_0"] = tf.Variable(
            np.random.normal(loc=0, scale=glorot, size=(input_size, self.deep_layers[0])), dtype=np.float32)
        weights["bias_0"] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(1, self.deep_layers[0])),
                                                        dtype=np.float32)  # 1 * layers[0]
        for i in range(1, num_layer):
            glorot = np.sqrt(2.0 / (self.deep_layers[i-1] + self.deep_layers[i]))
            weights["layer_%d" % i] = tf.Variable(
                np.random.normal(loc=0, scale=glorot, size=(self.deep_layers[i-1], self.deep_layers[i])),
                dtype=np.float32)  # layers[i-1] * layers[i]
            weights["bias_%d" % i] = tf.Variable(
                np.random.normal(loc=0, scale=glorot, size=(1, self.deep_layers[i])),
                dtype=np.float32)  # 1 * layer[i]

        # final concat projection layer
        if self.use_fm and self.use_deep:
            input_size = self.field_size + self.embedding_size + self.deep_layers[-1]
        elif self.use_fm:
            input_size = self.field_size + self.embedding_size
        elif self.use_deep:
            input_size = self.deep_layers[-1]
        glorot = np.sqrt(2.0 / (input_size + 1))
        weights["concat_projection"] = tf.Variable(
                        np.random.normal(loc=0, scale=glorot, size=(input_size, 1)),
                        dtype=np.float32)  # layers[i-1]*layers[i]
        weights["concat_bias"] = tf.Variable(tf.constant(0.01), dtype=np.float32)

        return weights


    def batch_norm_layer(self, x, train_phase, scope_bn):
        bn_train = batch_norm(x, decay=self.batch_norm_decay, center=True, scale=True, updates_collections=None,
                              is_training=True, reuse=None, trainable=True, scope=scope_bn)
        bn_inference = batch_norm(x, decay=self.batch_norm_decay, center=True, scale=True, updates_collections=None,
                                  is_training=False, reuse=True, trainable=True, scope=scope_bn)
        z = tf.cond(train_phase, lambda: bn_train, lambda: bn_inference)
        return z


    def get_batch(self, Xi, Xv, y, batch_size, index):
        start = index * batch_size
        end = (index+1) * batch_size
        end = end if end < len(y) else len(y)
        return Xi[start:end], Xv[start:end], [[y_] for y_ in y[start:end]]


    # shuffle three lists simutaneously
    def shuffle_in_unison_scary(self, a, b, c):
        rng_state = np.random.get_state()
        np.random.shuffle(a)
        np.random.set_state(rng_state)
        np.random.shuffle(b)
        np.random.set_state(rng_state)
        np.random.shuffle(c)


    def fit_on_batch(self, Xi, Xv, y):
        feed_dict = {self.feat_index: Xi,
                     self.feat_value: Xv,
                     self.label: y,
                     self.dropout_keep_fm: self.dropout_fm,
                     self.dropout_keep_deep: self.dropout_deep,
                     self.train_phase: True}
        loss, opt = self.sess.run((self.loss, self.optimizer), feed_dict=feed_dict)
        return loss


    def fit(self, Xi_train, Xv_train, y_train,
            Xi_valid=None, Xv_valid=None, y_valid=None,
            early_stopping=False, refit=False):
        """
        :param Xi_train: [[ind1_1, ind1_2, ...], [ind2_1, ind2_2, ...], ..., [indi_1, indi_2, ..., indi_j, ...], ...]
                         indi_j is the feature index of feature field j of sample i in the training set
        :param Xv_train: [[val1_1, val1_2, ...], [val2_1, val2_2, ...], ..., [vali_1, vali_2, ..., vali_j, ...], ...]
                         vali_j is the feature value of feature field j of sample i in the training set
                         vali_j can be either binary (1/0, for binary/categorical features) or float (e.g., 10.24, for numerical features)
        :param y_train: label of each sample in the training set
        :param Xi_valid: list of list of feature indices of each sample in the validation set
        :param Xv_valid: list of list of feature values of each sample in the validation set
        :param y_valid: label of each sample in the validation set
        :param early_stopping: perform early stopping or not
        :param refit: refit the model on the train+valid dataset or not
        :return: None
        """
        has_valid = Xv_valid is not None
        for epoch in range(self.epoch):
            t1 = time()
            self.shuffle_in_unison_scary(Xi_train, Xv_train, y_train)
            total_batch = int(len(y_train) / self.batch_size)
            for i in range(total_batch):
                Xi_batch, Xv_batch, y_batch = self.get_batch(Xi_train, Xv_train, y_train, self.batch_size, i)
                self.fit_on_batch(Xi_batch, Xv_batch, y_batch)

            # evaluate training and validation datasets
            train_result = self.evaluate(Xi_train, Xv_train, y_train)
            self.train_result.append(train_result)
            if has_valid:
                valid_result = self.evaluate(Xi_valid, Xv_valid, y_valid)
                self.valid_result.append(valid_result)
            if self.verbose > 0 and epoch % self.verbose == 0:
                if has_valid:
                    print("[%d] train-result=%.4f, valid-result=%.4f [%.1f s]"
                        % (epoch + 1, train_result, valid_result, time() - t1))
                else:
                    print("[%d] train-result=%.4f [%.1f s]"
                        % (epoch + 1, train_result, time() - t1))
            if has_valid and early_stopping and self.training_termination(self.valid_result):
                break

        # fit a few more epoch on train+valid until result reaches the best_train_score
        if has_valid and refit:
            if self.greater_is_better:
                best_valid_score = max(self.valid_result)
            else:
                best_valid_score = min(self.valid_result)
            best_epoch = self.valid_result.index(best_valid_score)
            best_train_score = self.train_result[best_epoch]
            Xi_train = Xi_train + Xi_valid
            Xv_train = Xv_train + Xv_valid
            y_train = y_train + y_valid
            for epoch in range(100):
                self.shuffle_in_unison_scary(Xi_train, Xv_train, y_train)
                total_batch = int(len(y_train) / self.batch_size)
                for i in range(total_batch):
                    Xi_batch, Xv_batch, y_batch = self.get_batch(Xi_train, Xv_train, y_train,
                                                                self.batch_size, i)
                    self.fit_on_batch(Xi_batch, Xv_batch, y_batch)
                # check
                train_result = self.evaluate(Xi_train, Xv_train, y_train)
                if abs(train_result - best_train_score) < 0.001 or \
                    (self.greater_is_better and train_result > best_train_score) or \
                    ((not self.greater_is_better) and train_result < best_train_score):
                    break


    def training_termination(self, valid_result):
        if len(valid_result) > 5:
            if self.greater_is_better:
                if valid_result[-1] < valid_result[-2] and \
                    valid_result[-2] < valid_result[-3] and \
                    valid_result[-3] < valid_result[-4] and \
                    valid_result[-4] < valid_result[-5]:
                    return True
            else:
                if valid_result[-1] > valid_result[-2] and \
                    valid_result[-2] > valid_result[-3] and \
                    valid_result[-3] > valid_result[-4] and \
                    valid_result[-4] > valid_result[-5]:
                    return True
        return False


    def predict(self, Xi, Xv):
        """
        :param Xi: list of list of feature indices of each sample in the dataset
        :param Xv: list of list of feature values of each sample in the dataset
        :return: predicted probability of each sample
        """
        # dummy y
        dummy_y = [1] * len(Xi)
        batch_index = 0
        Xi_batch, Xv_batch, y_batch = self.get_batch(Xi, Xv, dummy_y, self.batch_size, batch_index)
        y_pred = None
        while len(Xi_batch) > 0:
            num_batch = len(y_batch)
            feed_dict = {self.feat_index: Xi_batch,
                         self.feat_value: Xv_batch,
                         self.label: y_batch,
                         self.dropout_keep_fm: [1.0] * len(self.dropout_fm),
                         self.dropout_keep_deep: [1.0] * len(self.dropout_deep),
                         self.train_phase: False}
            batch_out = self.sess.run(self.out, feed_dict=feed_dict)

            if batch_index == 0:
                y_pred = np.reshape(batch_out, (num_batch,))
            else:
                y_pred = np.concatenate((y_pred, np.reshape(batch_out, (num_batch,))))

            batch_index += 1
            Xi_batch, Xv_batch, y_batch = self.get_batch(Xi, Xv, dummy_y, self.batch_size, batch_index)

        return y_pred


    def evaluate(self, Xi, Xv, y):
        """
        :param Xi: list of list of feature indices of each sample in the dataset
        :param Xv: list of list of feature values of each sample in the dataset
        :param y: label of each sample in the dataset
        :return: metric of the evaluation
        """
        y_pred = self.predict(Xi, Xv)
        iso_auc = roc_auc_score(y, y_pred)
        print('auc sorce {}'.format(iso_auc))
        return self.eval_metric(y, y_pred)

class GetFeature_V1(object):
    def __init__(self, historical_transactions_f, new_merchant_transactions_f, train_f, test_f, merchants_f):
        self.dataroot = "../data/"
        self.cacheRoot = "../cache/"
        self.submitRoot = "../submit/"
        self.inDirDict = {"data":self.dataroot, "cache":self.cacheRoot, "sub":self.submitRoot }
        self.his_f = self.dataroot + historical_transactions_f
        self.new_f = self.dataroot + new_merchant_transactions_f
        self.mer_f = self.dataroot + merchants_f
        self.train_f = self.dataroot  + train_f
        self.test_f = self.dataroot + test_f
        self.data = pd.DataFrame()
        self.transactions = self.read_mecherants(self.his_f)
        self.new_transactions = self.read_mecherants(self.new_f)
    def read_trn_tst(self,fname):
        df = self.readFile(fname,"data")
        df = df.fillna('2017-03')
        df['first_active_month_dt'] = pd.to_datetime(df['first_active_month'])
        df['elapsed_days'] = (  datetime.date(2018, 2, 1) - df['first_active_month_dt'].dt.date ).dt.days
        df['elapsed_years'] = 2018 - df['first_active_month_dt'].dt.year
        df['elapsed_months'] = round((  datetime.date(2018, 2, 1) - df['first_active_month_dt'].dt.date ).dt.days/30)
        return df
    def readFile(self, fname, inDirType ):
        df =  self.reduce_mem_usage(pd.read_csv(self.inDirDict[inDirType] + fname))
        return df
    def read_mecherants(self, fname):
        df = self.readFile(fname,"data")
        # if 'new' in fname:
        #     pass
        # else:
        #     df['purchase_date_dt'] = pd.to_datetime(df['purchase_date'])
        #     df['purchase_date_dtstr'] = df['purchase_date'].map(lambda x: x[:10]) ## map holidays and count on day
        return df
    def clean_as_needed(self):
        self.transactions['category_2'].fillna(1.0, inplace=True)
        self.transactions['category_3'].fillna('A', inplace=True)
        self.new_transactions['category_2'].fillna(1.0, inplace=True)
        self.new_transactions['category_3'].fillna('A', inplace=True)
        


    def transactions_clean(self):
        trn = self.read_trn_tst(self.train_f)
        tsn = self.read_trn_tst(self.test_f)
        tmpdata = pd.concat([trn, tsn])
        curUseVersion = 'V1'
        VerExists = '../data/{}Feature.pkl'.format(curUseVersion)
        if os.path.exists(VerExists):
            self.data['card_id'] = tmpdata['card_id']
        else:
            self.data = tmpdata


        del trn, tsn
        print('get transaction ')
        print('cleaning old .... 1')
        scaler = MinMaxScaler()
        self.transactions['installment_is_neg1_999'] = self.transactions.installments.map( lambda x:1 if x in [-1,999] else 0 )
        self.transactions['purchase_amount'] = scaler.fit_transform(np.array(self.transactions.purchase_amount).reshape(-1, 1))
        self.new_transactions['purchase_amount'] = scaler.fit_transform(
            np.array(self.new_transactions.purchase_amount).reshape(-1, 1))
        self.transactions['installment_is_zero'] = self.transactions['installments'].map( lambda x: 1 if x==0 else 0)
        self.transactions['purchase_date'] = pd.to_datetime(self.transactions['purchase_date'])
        self.transactions['purchase_date_str'] = self.transactions['purchase_date'].map(lambda x: x.strftime('%Y-%m-%d'))
        self.transactions = self.transactions.sort_values(['purchase_date'])

        self.transactions["day_diff"] = (self.transactions.groupby("card_id")["purchase_date"].diff(periods=1)).dt.days
        self.transactions['authorized_flag'] = self.transactions['authorized_flag'].map({'Y': 1, 'N': 0})



        self.transactions['neg_authorized_flag'] = 1- self.transactions['authorized_flag']
        self.transactions['category_1'] = self.transactions['category_1'].map({'Y': 1, 'N': 0})
        self.transactions['month'] = self.transactions['purchase_date'].dt.month
        self.transactions['dayofweek'] = self.transactions['purchase_date'].dt.dayofweek
        self.transactions['weekofyear'] = self.transactions['purchase_date'].dt.weekofyear
        self.transactions['quarter'] = self.transactions['purchase_date'].dt.quarter
        self.transactions['weekend'] = (self.transactions.purchase_date.dt.weekday >= 5).astype(int)
        self.transactions['weekend_purchase_amount'] = self.transactions['weekend'] * self.transactions['purchase_amount']
        self.transactions['hour'] = self.transactions['purchase_date'].dt.hour
        self.transactions['hourMinutes'] = self.transactions['hour'] * 60 + self.transactions['purchase_date'].dt.minute
        self.transactions['is_month_start'] = self.transactions['purchase_date'].dt.is_month_start
        self.transactions['month_diff'] = ((pd.to_datetime('2018-03-01')  - self.transactions[
            'purchase_date']).dt.days) // 30
        self.transactions['las_k_month'] = ((pd.to_datetime('2018-03-01') - self.transactions[
            'purchase_date']).dt.days) // 30
        self.transactions['month_diff'] += self.transactions['month_lag']

        # date 20190123------------------------------

        # impute missing values - This is now excluded.
        self.transactions['category_2'].fillna(1.0, inplace=True)
        self.transactions['category_3'].fillna('A', inplace=True)
        # self.transactions['category_3'] = self.transactions['category_3'].map({'A': 0, 'B': 1, 'C': 2})
        # another version by one hot

        # drive back score
        # self.transactions = pd.get_dummies( self.transactions, columns=['category_2', 'category_3'])
        # self.transactions['pay_category_2_1'] = self.transactions['category_2_1.0'] * self.transactions['purchase_amount']
        # self.transactions['pay_category_2_2'] = self.transactions['category_2_2.0'] * self.transactions['purchase_amount']
        # self.transactions['pay_category_2_3'] = self.transactions['category_2_3.0'] * self.transactions['purchase_amount']
        # self.transactions['pay_category_2_4'] = self.transactions['category_2_4.0'] * self.transactions['purchase_amount']
        # self.transactions['pay_category_3_A'] = self.transactions['category_3_A'] * self.transactions['purchase_amount']
        # self.transactions['pay_category_3_B'] = self.transactions['category_3_B'] * self.transactions['purchase_amount']
        # date 20190123------------------------------

        self.transactions['merchant_id'].fillna('M_ID_00a6ca8a8a', inplace=True)
        print('cleaning new .... 2')


        self.new_transactions['installment_is_neg1_999'] = self.new_transactions.installments.map(
            lambda x: 1 if x in [-1, 999] else 0)
        self.new_transactions['authorized_flag'] = self.new_transactions['authorized_flag'].map({'Y': 1, 'N': 0})
        self.new_transactions['category_1'] = self.new_transactions['category_1'].map({'Y': 1, 'N': 0})
        self.new_transactions['purchase_date'] = pd.to_datetime(self.new_transactions['purchase_date'])
        self.new_transactions['year'] = self.new_transactions['purchase_date'].dt.year
        self.new_transactions['weekofyear'] = self.new_transactions['purchase_date'].dt.weekofyear
        self.new_transactions['month'] = self.new_transactions['purchase_date'].dt.month
        self.new_transactions['dayofweek'] = self.new_transactions['purchase_date'].dt.dayofweek
        self.new_transactions['weekend'] = (self.new_transactions.purchase_date.dt.weekday >= 5).astype(int)
        self.new_transactions['hour'] = self.new_transactions['purchase_date'].dt.hour
        self.new_transactions['quarter'] = self.new_transactions['purchase_date'].dt.quarter
        self.new_transactions['is_month_start'] = self.new_transactions['purchase_date'].dt.is_month_start

        self.new_transactions['month_diff'] = ((pd.to_datetime('2018-05-01') - self.new_transactions[
            'purchase_date']).dt.days) // 30
        self.new_transactions['month_diff'] += self.new_transactions['month_lag']

        # impute missing values
        self.new_transactions['category_2'].fillna(1.0, inplace=True)
        self.new_transactions['category_3'].fillna('A', inplace=True)

    def get_cate_feats(self):
        new_cates = []
        cate_2_vals = self.transactions.category_2.unique().tolist()
        for cate2v in cate_2_vals:
            gp = self.new_transactions[ self.new_transactions.category_2 == cate2v ].groupby('card_id')['card_id'].count()
            self.data[ 'new_category_2_val_{}_count'.format( cate2v ) ] = self.data['card_id'].map( gp )
            self.data['new_category_2_val_{}_count'.format(cate2v)] = self.data[ 'new_category_2_val_{}_count'.format( cate2v ) ].fillna(0)
            new_cates.append( 'new_category_2_val_{}_count'.format(cate2v) )
            gp = self.transactions[ self.transactions.category_2 == cate2v ].groupby('card_id')['card_id'].count()
            self.data['his_category_2_val_{}_count'.format(cate2v)] = self.data['card_id'].map(gp)
            self.data['his_category_2_val_{}_count'.format(cate2v)] = self.data[
                'his_category_2_val_{}_count'.format(cate2v)].fillna(0)
            new_cates.append( 'his_category_2_val_{}_count'.format(cate2v) )
        cate_3_vals = self.transactions.category_3.unique().tolist()
        for cate3v in cate_3_vals:
            gp = self.new_transactions[self.new_transactions.category_3 == cate3v].groupby('card_id')['card_id'].count()
            self.data['new_category_3_val_{}_count'.format(cate3v)] = self.data['card_id'].map(gp)
            self.data['new_category_3_val_{}_count'.format(cate3v)] = self.data[
                'new_category_3_val_{}_count'.format(cate3v)].fillna(0)
            new_cates.append( 'new_category_3_val_{}_count'.format(cate3v) )
            gp = self.transactions[self.transactions.category_3 == cate3v].groupby('card_id')['card_id'].count()
            self.data['his_category_3_val_{}_count'.format(cate3v)] = self.data['card_id'].map(gp)
            self.data['his_category_3_val_{}_count'.format(cate3v)] = self.data[
                'his_category_3_val_{}_count'.format(cate3v)].fillna(0)
            new_cates.append( 'his_category_3_val_{}_count'.format(cate3v) )
            return new_cates


        # self.new_transactions['category_3'] = self.new_transactions['category_3'].map({'A': 0, 'B': 1, 'C': 2})

        # self.new_transactions = pd.get_dummies(self.new_transactions, columns=['category_2', 'category_3'])
        # self.new_transactions['pay_category_2_1'] = self.new_transactions['category_2_1.0'] * self.new_transactions['purchase_amount']
        # self.new_transactions['pay_category_2_2'] = self.new_transactions['category_2_2.0'] * self.new_transactions['purchase_amount']
        # self.new_transactions['pay_category_2_3'] = self.new_transactions['category_2_3.0'] * self.new_transactions['purchase_amount']
        # self.new_transactions['pay_category_2_4'] = self.new_transactions['category_2_4.0'] * self.new_transactions['purchase_amount']
        # self.new_transactions['pay_category_3_A'] = self.new_transactions['category_3_A'] * self.new_transactions['purchase_amount']
        # self.new_transactions['pay_category_3_B'] = self.new_transactions['category_3_B'] * self.new_transactions['purchase_amount']

        # self.new_transactions['merchant_id'].fillna('M_ID_00a6ca8a8a',inplace=True)


        # not a prover
        # self.transactions['purchase_amount_cumsum'] = self.transactions.groupby(['card_id', 'merchant_id'])[
        #     'purchase_amount'].cumsum()
        # self.transactions['authorized_flag_cumsum'] = self.transactions.groupby(['card_id', 'merchant_id'])[
        #     'authorized_flag'].cumsum()
        # self.transactions['category_1_cumsum'] = self.transactions.groupby(['card_id', 'merchant_id'])[
        #     'category_1'].cumsum()
        # self.transactions['purchase_amount_cateID_cumsum'] = self.transactions.groupby(['card_id','merchant_category_id'])[
        #     'purchase_amount'].cumsum()


    def reduce_mem_usage(self, df, verbose=True):
        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        start_mem = df.memory_usage().sum() / 1024 ** 2
        for col in df.columns:
            col_type = df[col].dtypes
            if col_type in numerics:
                c_min = df[col].min()
                c_max = df[col].max()
                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)
                else:
                    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        df[col] = df[col].astype(np.float16)
                    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
                    else:
                        df[col] = df[col].astype(np.float64)
        end_mem = df.memory_usage().sum() / 1024 ** 2
        if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (
                    start_mem - end_mem) / start_mem))
        return df
    



    def getFea(self):
        print('loading ... ')
        lasUseVersion = 'V6-6-1'
        VerExists = '../data/{}Feature.pkl'.format(lasUseVersion)
        if os.path.exists(VerExists):
            print('read existing features .. ')
            self.data = pd.read_pickle(VerExists)
            self.clean_as_needed()
            new_cates = self.get_cate_feats()
            # self.clean_as_needed()
            # self.from_new_tohis_recommed()
        else:
            super().getFeaV6()
            lasIsStable = True  # 手工设置确认
            if lasIsStable:
                self.data.to_pickle(VerExists)
            new_cates = self.get_cate_feats()
            # self.from_new_tohis_recommed()
        return new_cates
            
historical_transactions_f = "historical_transactions.csv"
new_merchant_transactions_f = 'new_merchant_transactions.csv'
train_f =  'train.csv'
test_f = 'test.csv'
merchants_f = 'merchants.csv'
getFeature = GetFeature_V1( historical_transactions_f, new_merchant_transactions_f, train_f, test_f, merchants_f)
new_cates = getFeature.getFea()
cate_feas = ['elapsed_months'
,'elapsed_years'
,'feature_1'
,'feature_2'
,'feature_3'
,'new_transactions_count'
,'new_Valentine_Day_2017_mean'
,'new_installment_is_neg1_999_sum'
,'new_installment_is_neg1_999_mean'
,'new_Valentine_Day_2017_mean'
,'new_installment_is_neg1_999_sum'
,'new_installment_is_neg1_999_mean'
,'new_dayofweek_nunique'
,'new_city_id_nunique'
,'new_authorized_flag_sum'
,'new_card_id_size'
,'new_installments_sum'
,'new_category_1_sum'
,'hist_city_id_nunique'
,'hist_authorized_flag_sum'
,'hist_weekend_sum'
,'hist_installments_sum'
,'hist_category_1_sum'
,'hist_state_id_nunique'
,'Lag_3installment_is_zero_sum'
,'Lag_3installment_is_neg1_999_sum'
,'Lag_3installment_is_neg1_999_mean']



data = getFeature.data
data = data.drop(['first_active_month', 'first_active_month_dt'] ,axis=1)



for col in data.columns:
    if col in ['card_id','target']:
        continue
    colvals = data[col].map(str).unique().tolist()
    msg = ''
    if 'inf' in colvals:
        msg += '{} contains inf '.format( col )
    if 'nan' in colvals:
        msg +=  '{} contains nan '.format( col )
    if len(msg)>0:
        print(msg)
        col_mode = data[col].value_counts().index[0]

        if 'inf' in str(col_mode):
            print(col_mode)
            col_mode = data[col].value_counts().index[1]
            print('after process inf :{}'.format( col_mode ))
        else:
            print(col_mode)

        data[col] = data[col].map(lambda x: col_mode if str(x) == 'nan' else x)
        vls = list(set(list(data[col])))
        col_max = max([vl for vl in vls if 'inf' not in str(vl)])
        col_min = min([vl for vl in vls if 'inf' not in str(vl)])
        del vls
        data[col] = data[col].map(lambda x: col_max if str(x) == 'inf' else (col_min if str(x) == '-inf' else x )   )







cate_feas = cate_feas + new_cates

numberics_feats = [c for c in data.columns if c not in cate_feas + ['card_id','target'] ]

class config(object):

    SUB_DIR = "../submit"

    NUM_SPLITS = 3
    RANDOM_SEED = 2018

    # types of columns of the dataset dataframe
    CATEGORICAL_COLS = cate_feas

    NUMERIC_COLS = numberics_feats

    IGNORE_COLS = [ "card_id", "target"]
    

import pandas as pd


class FeatureDictionary(object):
    def __init__(self, trainfile=None, testfile=None,
                 dfTrain=None, dfTest=None, numeric_cols=[], ignore_cols=[]):
        assert not ((trainfile is None) and (dfTrain is None)), "trainfile or dfTrain at least one is set"
        assert not ((trainfile is not None) and (dfTrain is not None)), "only one can be set"
        assert not ((testfile is None) and (dfTest is None)), "testfile or dfTest at least one is set"
        assert not ((testfile is not None) and (dfTest is not None)), "only one can be set"
        self.trainfile = trainfile
        self.testfile = testfile
        self.dfTrain = dfTrain
        self.dfTest = dfTest
        self.numeric_cols = numeric_cols
        self.ignore_cols = ignore_cols
        self.gen_feat_dict()

    def gen_feat_dict(self):
        if self.dfTrain is None:
            dfTrain = pd.read_csv(self.trainfile)
        else:
            dfTrain = self.dfTrain
        if self.dfTest is None:
            dfTest = pd.read_csv(self.testfile)
        else:
            dfTest = self.dfTest
        df = pd.concat([dfTrain, dfTest])
        self.feat_dict = {}
        tc = 0
        for col in df.columns:
            if col in self.ignore_cols:
                continue
            if col in self.numeric_cols:
                # map to a single index
                self.feat_dict[col] = tc
                tc += 1
            else:
                us = df[col].unique()
                self.feat_dict[col] = dict(zip(us, range(tc, len(us)+tc)))
                tc += len(us)
        self.feat_dim = tc


class DataParser(object):
    def __init__(self, feat_dict):
        self.feat_dict = feat_dict

    def parse(self, infile=None, df=None, has_label=False):
        assert not ((infile is None) and (df is None)), "infile or df at least one is set"
        assert not ((infile is not None) and (df is not None)), "only one can be set"
        if infile is None:
            dfi = df.copy()
        else:
            dfi = pd.read_csv(infile)
        if has_label:
            y = dfi["target"].map(lambda x: 1 if x<= -17 else 0)
            y = y.values.tolist()
            dfi.drop(["card_id", "target"], axis=1, inplace=True)
        else:
            ids = dfi["card_id"].values.tolist()
            dfi.drop(["card_id"], axis=1, inplace=True)
        # dfi for feature index
        # dfv for feature value which can be either binary (1/0) or float (e.g., 10.24)
        dfv = dfi.copy()
        for col in dfi.columns:
            if col in self.feat_dict.ignore_cols:
                dfi.drop(col, axis=1, inplace=True)
                dfv.drop(col, axis=1, inplace=True)
                continue
            if col in self.feat_dict.numeric_cols:
                dfi[col] = self.feat_dict.feat_dict[col]
            else:
                dfi[col] = dfi[col].map(self.feat_dict.feat_dict[col])
                dfv[col] = 1.

        # list of list of feature indices of each sample in the dataset
        Xi = dfi.values.tolist()
        # list of list of feature values of each sample in the dataset
        Xv = dfv.values.tolist()
        if has_label:
            return Xi, Xv, y
        else:
            return Xi, Xv, ids
            
import numpy as np

def gini(actual, pred):
    assert (len(actual) == len(pred))
    all = np.asarray(np.c_[actual, pred, np.arange(len(actual))], dtype=np.float)
    all = all[np.lexsort((all[:, 2], -1 * all[:, 1]))]
    totalLosses = all[:, 0].sum()
    giniSum = all[:, 0].cumsum().sum() / totalLosses

    giniSum -= (len(actual) + 1) / 2.
    return giniSum / len(actual)

def gini_norm(actual, pred):
    return gini(actual, pred) / gini(actual, actual)


import os
import sys

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import make_scorer
from sklearn.model_selection import StratifiedKFold



gini_scorer = make_scorer(gini_norm, greater_is_better=True, needs_proba=True)


def _load_data(data):

    dfTrain = data[ data.target.isnull() == False ]
    dfTest = data[ data.target.isnull() == True ]

#     def preprocess(df):
#         cols = [c for c in df.columns if c not in ["card_id", "target"]]
#         df["missing_feat"] = np.sum((df[cols] == -1).values, axis=1)
#         df["ps_car_13_x_ps_reg_03"] = df["ps_car_13"] * df["ps_reg_03"]
#         return df

#     dfTrain = preprocess(dfTrain)
#     dfTest = preprocess(dfTest)

    cols = [c for c in dfTrain.columns if c not in ["card_id", "target"]]
    cols = [c for c in cols if (not c in config.IGNORE_COLS)]

    X_train = dfTrain[cols].values
    y_train = dfTrain["target"].map(lambda x: 1 if x<= -17 else 0)
    y_train = y_train.values
    
    
    X_test = dfTest[cols].values
    ids_test = dfTest["card_id"].values
    cat_features_indices = [i for i,c in enumerate(cols) if c in config.CATEGORICAL_COLS]

    return dfTrain, dfTest, X_train, y_train, X_test, ids_test, cat_features_indices

def _make_submission(ids, y_pred, filename="submission.csv"):
    pd.DataFrame({"card_id": ids, "target": y_pred.flatten()}).to_csv(
        os.path.join(config.SUB_DIR, filename), index=False, float_format="%.5f")
def _run_base_model_dfm(dfTrain, dfTest, folds, dfm_params):
    fd = FeatureDictionary(dfTrain=dfTrain, dfTest=dfTest,
                           numeric_cols=config.NUMERIC_COLS,
                           ignore_cols=config.IGNORE_COLS)
    data_parser = DataParser(feat_dict=fd)
    Xi_train, Xv_train, y_train = data_parser.parse(df=dfTrain, has_label=True)
    Xi_test, Xv_test, ids_test = data_parser.parse(df=dfTest)

    dfm_params["feature_size"] = fd.feat_dim
    dfm_params["field_size"] = len(Xi_train[0])

    y_train_meta = np.zeros((dfTrain.shape[0], 1), dtype=float)
    y_test_meta = np.zeros((dfTest.shape[0], 1), dtype=float)
    _get = lambda x, l: [x[i] for i in l]
    gini_results_cv = np.zeros(len(folds), dtype=float)
    gini_results_epoch_train = np.zeros((len(folds), dfm_params["epoch"]), dtype=float)
    gini_results_epoch_valid = np.zeros((len(folds), dfm_params["epoch"]), dtype=float)
    for i, (train_idx, valid_idx) in enumerate(folds):
        Xi_train_, Xv_train_, y_train_ = _get(Xi_train, train_idx), _get(Xv_train, train_idx), _get(y_train, train_idx)
        Xi_valid_, Xv_valid_, y_valid_ = _get(Xi_train, valid_idx), _get(Xv_train, valid_idx), _get(y_train, valid_idx)

        dfm = DeepFM(**dfm_params)
        dfm.fit(Xi_train_, Xv_train_, y_train_, Xi_valid_, Xv_valid_, y_valid_)

        y_train_meta[valid_idx,0] = dfm.predict(Xi_valid_, Xv_valid_)
        y_test_meta[:,0] += dfm.predict(Xi_test, Xv_test)

        gini_results_cv[i] = gini_norm(y_valid_, y_train_meta[valid_idx])
        gini_results_epoch_train[i] = dfm.train_result
        gini_results_epoch_valid[i] = dfm.valid_result

    y_test_meta /= float(len(folds))

    # save result
    if dfm_params["use_fm"] and dfm_params["use_deep"]:
        clf_str = "DeepFM"
    elif dfm_params["use_fm"]:
        clf_str = "FM"
    elif dfm_params["use_deep"]:
        clf_str = "DNN"
    print("%s: %.5f (%.5f)"%(clf_str, gini_results_cv.mean(), gini_results_cv.std()))
    filename = "%s_Mean%.5f_Std%.5f.csv"%(clf_str, gini_results_cv.mean(), gini_results_cv.std())
    _make_submission(ids_test, y_test_meta, filename)

#     _plot_fig(gini_results_epoch_train, gini_results_epoch_valid, clf_str)

    return y_train_meta, y_test_meta





# def _plot_fig(train_results, valid_results, model_name):
#     colors = ["red", "blue", "green"]
#     xs = np.arange(1, train_results.shape[1]+1)
#     plt.figure()
#     legends = []
#     for i in range(train_results.shape[0]):
#         plt.plot(xs, train_results[i], color=colors[i], linestyle="solid", marker="o")
#         plt.plot(xs, valid_results[i], color=colors[i], linestyle="dashed", marker="o")
#         legends.append("train-%d"%(i+1))
#         legends.append("valid-%d"%(i+1))
#     plt.xlabel("Epoch")
#     plt.ylabel("Normalized Gini")
#     plt.title("%s"%model_name)
#     plt.legend(legends)
#     plt.savefig("./fig/%s.png"%model_name)
#     plt.close()


# load data
dfTrain, dfTest, X_train, y_train, X_test, ids_test, cat_features_indices = _load_data(data)

# folds
folds = list(StratifiedKFold(n_splits=config.NUM_SPLITS, shuffle=True,
                             random_state=config.RANDOM_SEED).split(X_train, y_train))


# ------------------ DeepFM Model ------------------
# params
dfm_params = {
    "use_fm": True,
    "use_deep": True,
    "embedding_size": 8,
    "dropout_fm": [1.0, 1.0],
    "deep_layers": [32, 32],
    "dropout_deep": [0.5, 0.5, 0.5],
    "deep_layers_activation": tf.nn.relu,
    "epoch": 30,
    "batch_size": 1024,
    "learning_rate": 0.001,
    "optimizer_type": "adam",
    "batch_norm": 1,
    "batch_norm_decay": 0.995,
    "l2_reg": 0.01,
    "verbose": True,
    "eval_metric": gini_norm,
    "random_seed": config.RANDOM_SEED
}
y_train_dfm, y_test_dfm = _run_base_model_dfm(dfTrain, dfTest, folds, dfm_params)

print(list(y_test_dfm)[:100] )