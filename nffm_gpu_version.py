import os
import time
import pandas as pd
os.system('pip install tensorflow-gpu')
#os.system('pip install xgboost -gpu')
# os.system('pip install tqdm')

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




import abc
import math
import tensorflow as tf
from sklearn import metrics
import os

import numpy as np
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm
import tensorflow as tf
from tensorflow.python.ops import lookup_ops
from tensorflow.python.layers import core as layers_core
import numpy as np
import time 
import os

import codecs
import collections
import json
import math
import os
import sys
import time

import numpy as np
import tensorflow as tf
import pandas as pd
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import os
import gc



def hash_batch(batch,hparams):
    batch=pd.DataFrame(batch)
    batch=list(batch.values)
    for b in batch:
        for i in range(len(b)):
            b[i]=abs(hash('key_'+str(i)+' value_'+str(b[i]))) % hparams.hash_ids
    return batch

def print_time(s, start_time):
  """Take a start time, print elapsed duration, and return a new time."""
  print("%s, time %ds, %s." % (s, (time.time() - start_time), time.ctime()))
  sys.stdout.flush()
  return time.time()

def print_out(s, f=None, new_line=True):
  """Similar to print but with support to flush and output to a file."""
  if isinstance(s, bytes):
    s = s.decode("utf-8")

  if f:
    f.write(s.encode("utf-8"))
    if new_line:
      f.write(b"\n")

  # stdout
  out_s = s.encode("utf-8")
  if not isinstance(out_s, str):
    out_s = out_s.decode("utf-8")
  print(out_s, end="", file=sys.stdout)

  if new_line:
    sys.stdout.write("\n")
  sys.stdout.flush()

def print_step_info(prefix,epoch, global_step, info):
    print_out("%sepoch %d step %d lr %g logloss %.6f gN %.2f, %s" %
      (prefix, epoch,global_step, info["learning_rate"],
       info["train_ppl"], info["avg_grad_norm"], time.ctime())) 
    
def print_hparams(hparams, skip_patterns=None, header=None):
  """Print hparams, can skip keys based on pattern."""
  if header: print_out("%s" % header)
  values = hparams.values()
  for key in sorted(values.keys()):
    if not skip_patterns or all(
        [skip_pattern not in key for skip_pattern in skip_patterns]):
      print_out("  %s=%s" % (key, str(values[key])))







class BaseModel(object):
    def __init__(self, hparams,  scope=None):
        tf.set_random_seed(1234)
        self.iterator = iterator
        self.layer_params = []
        self.embed_params = []
        self.cross_params = []
        self.layer_keeps = None
        self.keep_prob_train = None
        self.keep_prob_test = None
        self.initializer = self._get_initializer(hparams)
        self.logit = self._build_graph(hparams)
        self.pred = self._get_pred(self.logit, hparams)
        self.data_loss = self._compute_data_loss(hparams)
        self.regular_loss = self._compute_regular_loss(hparams)
        self.loss = tf.add(self.data_loss, self.regular_loss)
        self.saver = tf.train.Saver(max_to_keep=hparams.epochs)
        self.update = self._build_train_opt(hparams)
        self.init_op = tf.global_variables_initializer()
        self.merged = self._add_summaries()

    def _get_pred(self, logit, hparams):
        if hparams.method == 'regression':
            pred = tf.identity(logit)
        elif hparams.method == 'classification':
            pred = tf.sigmoid(logit)
        else:
            raise ValueError("method must be regression or classification, but now is {0}".format(hparams.method))
        return pred

    def _add_summaries(self):
        tf.summary.scalar("data_loss", self.data_loss)
        tf.summary.scalar("regular_loss", self.regular_loss)
        tf.summary.scalar("loss", self.loss)
        merged = tf.summary.merge_all()
        return merged

    @abc.abstractmethod
    def _build_graph(self, hparams):
        """Subclass must implement this."""
        pass

    def _l2_loss(self, hparams):
        l2_loss = tf.zeros([1], dtype=tf.float32)
        # embedding_layer l2 loss
        for param in self.embed_params:
            l2_loss = tf.add(l2_loss, tf.multiply(hparams.l2, tf.nn.l2_loss(param)))

        return l2_loss

    def _l1_loss(self, hparams):
        l1_loss = tf.zeros([1], dtype=tf.float32)
        # embedding_layer l2 loss
        for param in self.embed_params:
            l1_loss = tf.add(l1_loss, tf.multiply(hparams.embed_l1, tf.norm(param, ord=1)))
        params = self.layer_params
        for param in params:
            l1_loss = tf.add(l1_loss, tf.multiply(hparams.layer_l1, tf.norm(param, ord=1)))
        return l1_loss

    def _cross_l_loss(self, hparams):
        cross_l_loss = tf.zeros([1], dtype=tf.float32)
        for param in self.cross_params:
            cross_l_loss = tf.add(cross_l_loss, tf.multiply(hparams.cross_l1, tf.norm(param, ord=1)))
            cross_l_loss = tf.add(cross_l_loss, tf.multiply(hparams.cross_l2, tf.norm(param, ord=1)))
        return cross_l_loss 

    def _get_initializer(self, hparams):
        if hparams.init_method == 'tnormal':
            return tf.truncated_normal_initializer(stddev=hparams.init_value)
        elif hparams.init_method == 'uniform':
            return tf.random_uniform_initializer(-hparams.init_value, hparams.init_value)
        elif hparams.init_method == 'normal':
            return tf.random_normal_initializer(stddev=hparams.init_value)
        elif hparams.init_method == 'xavier_normal':
            return tf.contrib.layers.xavier_initializer(uniform=False)
        elif hparams.init_method == 'xavier_uniform':
            return tf.contrib.layers.xavier_initializer(uniform=True)
        elif hparams.init_method == 'he_normal':
            return tf.contrib.layers.variance_scaling_initializer( \
                factor=2.0, mode='FAN_AVG', uniform=False)
        elif hparams.init_method == 'he_uniform':
            return tf.contrib.layers.variance_scaling_initializer( \
                factor=2.0, mode='FAN_AVG', uniform=True)
        else:
            return tf.truncated_normal_initializer(stddev=hparams.init_value)

    def _compute_data_loss(self, hparams):
        if hparams.loss == 'cross_entropy_loss':
            data_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits( \
                logits=tf.reshape(self.logit, [-1]), \
                labels=tf.reshape(self.iterator.labels, [-1])))
        elif hparams.loss == 'square_loss':
            data_loss = tf.sqrt(tf.reduce_mean(
                tf.squared_difference(tf.reshape(self.pred, [-1]), tf.reshape(self.iterator.labels, [-1]))))
        elif hparams.loss == 'log_loss':
            data_loss = tf.reduce_mean(tf.losses.log_loss(predictions=tf.reshape(self.pred, [-1]),
                                                          labels=tf.reshape(self.iterator.labels, [-1])))
        else:
            raise ValueError("this loss not defined {0}".format(hparams.loss))
        return data_loss

    def _compute_regular_loss(self, hparams):
        regular_loss = self._l2_loss(hparams) + self._l1_loss(hparams) + self._cross_l_loss(hparams)
        regular_loss = tf.reduce_sum(regular_loss)
        return regular_loss

    def _build_train_opt(self, hparams):
        def train_opt(hparams):
            if hparams.optimizer == 'adadelta':
                train_step = tf.train.AdadeltaOptimizer( \
                    hparams.learning_rate)
            elif hparams.optimizer == 'adagrad':
                train_step = tf.train.AdagradOptimizer( \
                    hparams.learning_rate)
            elif hparams.optimizer == 'sgd':
                train_step = tf.train.GradientDescentOptimizer( \
                    hparams.learning_rate)
            elif hparams.optimizer == 'adam':
                train_step = tf.train.AdamOptimizer( \
                    hparams.learning_rate)
            elif hparams.optimizer == 'ftrl':
                train_step = tf.train.FtrlOptimizer( \
                    hparams.learning_rate)
            elif hparams.optimizer == 'gd':
                train_step = tf.train.GradientDescentOptimizer( \
                    hparams.learning_rate)
            elif hparams.optimizer == 'padagrad':
                train_step = tf.train.ProximalAdagradOptimizer( \
                    hparams.learning_rate)
            elif hparams.optimizer == 'pgd':
                train_step = tf.train.ProximalGradientDescentOptimizer( \
                    hparams.learning_rate)
            elif hparams.optimizer == 'rmsprop':
                train_step = tf.train.RMSPropOptimizer( \
                    hparams.learning_rate)
            else:
                train_step = tf.train.GradientDescentOptimizer( \
                    hparams.learning_rate)
            return train_step

        train_step = train_opt(hparams)
        return train_step
    
        
        
    def _active_layer(self, logit, scope, activation, layer_idx):
        logit = self._activate(logit, activation)
        return logit

    def _activate(self, logit, activation):
        if activation == 'sigmoid':
            return tf.nn.sigmoid(logit)
        elif activation == 'softmax':
            return tf.nn.softmax(logit)
        elif activation == 'relu':
            return tf.nn.relu(logit)
        elif activation == 'tanh':
            return tf.nn.tanh(logit)
        elif activation == 'elu':
            return tf.nn.elu(logit)
        elif activation == 'identity':
            return tf.identity(logit)
        else:
            raise ValueError("this activations not defined {0}".format(activation))

    def _dropout(self, logit, layer_idx):
        logit = tf.nn.dropout(x=logit, keep_prob=self.layer_keeps[layer_idx])
        return logit

    def train(self, sess):
        return sess.run([self.update, self.loss, self.data_loss, self.merged], \
                        feed_dict={self.layer_keeps: self.keep_prob_train})

    def eval(self,T,dev_data,hparams,sess):
        preds=self.infer(dev_data)
        if hparams.metric=='logloss':
            log_loss=metrics.log_loss(dev_data[1],preds)
            if self.best_score>log_loss:
                self.best_score=log_loss
                try:
                    os.makedirs('model_tmp/')
                except:
                    pass
                self.saver.save(sess,'model_tmp/model')
            print_out("# Epcho-time %.2fs Eval logloss %.6f. Best logloss %.6f." \
                            %(T,log_loss,self.best_score))
        elif hparams.metric=='auc':
            fpr, tpr, thresholds = metrics.roc_curve(dev_data[1]+1, preds, pos_label=2)
            auc=metrics.auc(fpr, tpr)
            if self.best_score<auc:
                self.best_score=auc
                try:
                    os.makedirs('model_tmp/')
                except:
                    pass
                self.saver.save(sess,'model_tmp/model')                           
            print_out("# Epcho-time %.2fs Eval AUC %.6f. Best AUC %.6f." \
                            %(T,auc,self.best_score))  

    def infer(self, sess):
        return sess.run([self.pred], \
                        feed_dict={self.layer_keeps: self.keep_prob_test})
    
    def batch_norm_layer(self, x, train_phase, scope_bn):
        bn_train = batch_norm(x, decay=self.hparams.batch_norm_decay, center=True, scale=True, updates_collections=None,is_training=True, reuse=None, trainable=True, scope=scope_bn)
        bn_inference = batch_norm(x, decay=self.hparams.batch_norm_decay, center=True, scale=True, updates_collections=None,is_training=False, reuse=True, trainable=True, scope=scope_bn)
        z = tf.cond(train_phase, lambda: bn_train, lambda: bn_inference)
        return z

class Model(BaseModel):
    def __init__(self,hparams):
        self.hparams=hparams
        if hparams.metric in ['logloss']:
            self.best_score=100000
        else:
            self.best_score=0
        self.build_graph(hparams)   
        self.optimizer(hparams)
        params = tf.trainable_variables()
        print_out("# Trainable variables")
        for param in params:
            print_out("  %s, %s, %s" % (param.name, str(param.get_shape()),param.op.device))   
  
    def set_Session(self,sess):
        self.sess=sess
        
    def build_graph(self, hparams):
        initializer = self._get_initializer(hparams)
        self.label = tf.placeholder(shape=(None), dtype=tf.float32)
        self.use_norm=tf.placeholder(tf.bool)
        self.features=tf.placeholder(shape=(None,hparams.feature_nums), dtype=tf.int32)
        self.emb_v1=tf.get_variable(shape=[hparams.hash_ids,1],
                                    initializer=initializer,name='emb_v1')
        self.emb_v2=tf.get_variable(shape=[hparams.hash_ids,hparams.feature_nums,hparams.k],
                                    initializer=initializer,name='emb_v2')
        
        #lr
        emb_inp_v1=tf.gather(self.emb_v1, self.features)
        w1=tf.reduce_sum(emb_inp_v1,[-1,-2])
        
        emb_inp_v2=tf.gather(self.emb_v2, self.features)
        emb_inp_v2=tf.reduce_sum(emb_inp_v2*tf.transpose(emb_inp_v2,[0,2,1,3]),-1)
        temp=[]
        for i in range(hparams.feature_nums):
            if i!=0:
                temp.append(emb_inp_v2[:,i,:i])
        w2=tf.reduce_sum(tf.concat(temp,-1),-1)
        
        #DNN
        dnn_input=tf.concat(temp,-1)
        input_size=int(dnn_input.shape[-1])
        for idx in range(len(hparams.hidden_size)):
            glorot = np.sqrt(2.0 / (input_size + hparams.hidden_size[idx]))
            W = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(input_size, hparams.hidden_size[idx])), dtype=np.float32)
            dnn_input=tf.tensordot(dnn_input,W,[[-1],[0]])
            if hparams.norm is True:
                dnn_input=self.batch_norm_layer(dnn_input,\
                                           self.use_norm,'norm_'+str(idx))
            dnn_input=tf.nn.relu(dnn_input)
            input_size=hparams.hidden_size[idx]

        glorot = np.sqrt(2.0 / (hparams.hidden_size[-1] + 1))
        W = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(hparams.hidden_size[-1], 1)), dtype=np.float32)
        b = tf.Variable(tf.constant(-3.5), dtype=np.float32)        
        w3=tf.tensordot(dnn_input,W,[[-1],[0]])+b 
        
        
        logit=w3[:,0]
        self.prob=tf.sigmoid(logit)
        logit_1=tf.log(self.prob+1e-20)
        logit_0=tf.log(1-self.prob+1e-20)
        self.loss=-tf.reduce_mean(self.label*logit_1+(1-self.label)*logit_0)
        self.cost=-(self.label*logit_1+(1-self.label)*logit_0)
        self.saver= tf.train.Saver()
            
    def optimizer(self,hparams):
        opt=self._build_train_opt(hparams)
        params = tf.trainable_variables()
        gradients = tf.gradients(self.loss,params,colocate_gradients_with_ops=True)
        clipped_grads, gradient_norm = tf.clip_by_global_norm(gradients, 5.0)  
        self.grad_norm =gradient_norm 
        self.update = opt.apply_gradients(zip(clipped_grads, params)) 

    def train(self,train_data,dev_data):
        hparams=self.hparams
        sess=self.sess
        assert len(train_data[0])==len(train_data[1]), "Size of features data must be equal to label"
        for epoch in range(hparams.epoch):
            info={}
            info['loss']=[]
            info['norm']=[]
            start_time = time.time()
            for idx in range(len(train_data[0])//hparams.batch_size+3):
                if idx*hparams.batch_size>=len(train_data[0]):
                    T=(time.time()-start_time)
                    self.eval(T,dev_data,hparams,sess)
                    break
                    
                batch=train_data[0][idx*hparams.batch_size:\
                                    min((idx+1)*hparams.batch_size,len(train_data[0]))]
                batch=hash_batch(batch,hparams)
                label=train_data[1][idx*hparams.batch_size:\
                                    min((idx+1)*hparams.batch_size,len(train_data[1]))]
                loss,_,norm=sess.run([self.loss,self.update,self.grad_norm],feed_dict=\
                                     {self.features:batch,self.label:label,self.use_norm:True})
                info['loss'].append(loss)
                info['norm'].append(norm)
                if (idx+1)%hparams.num_display_steps==0:
                    info['learning_rate']=hparams.learning_rate
                    info["train_ppl"]= np.mean(info['loss'])
                    info["avg_grad_norm"]=np.mean(info['norm'])
                    print_step_info("  ", epoch,idx+1, info)
                    del info
                    info={}
                    info['loss']=[]
                    info['norm']=[]
                if (idx+1)%hparams.num_eval_steps==0 and dev_data:
                    T=(time.time()-start_time)
                    self.eval(T,dev_data,hparams,sess)
        self.saver.restore(sess,'model_tmp/model')
        T=(time.time()-start_time)
        self.eval(T,dev_data,hparams,sess)
        os.system("rm -r model_tmp")
        
      
    def infer(self,dev_data):
        hparams=self.hparams
        sess=self.sess
        assert len(dev_data[0])==len(dev_data[1]), "Size of features data must be equal to label"       
        preds=[]
        total_loss=[]
        for idx in range(len(dev_data[0])//hparams.batch_size+1):
            batch=dev_data[0][idx*hparams.batch_size:\
                              min((idx+1)*hparams.batch_size,len(dev_data[0]))]
            batch=hash_batch(batch,hparams)
            label=dev_data[1][idx*hparams.batch_size:\
                              min((idx+1)*hparams.batch_size,len(dev_data[1]))]
            pred=sess.run(self.prob,feed_dict=\
                          {self.features:batch,self.label:label,self.use_norm:False})  
            preds.append(pred)   
        preds=np.concatenate(preds)
        return preds



# from models import fm
# from models import ffm
# from models import nffm
# import tensorflow as tf
# from imp import reload
def build_model(hparams):
    tf.reset_default_graph()
    model=Model(hparams)
    # if hparams.model=='fm':
    #     model=fm.Model(hparams)
    # elif hparams.model=='ffm':
    #     model=ffm.Model(hparams)
    # elif hparams.model=='nffm':
    #     model=nffm.Model(hparams)
    config_proto = tf.ConfigProto(log_device_placement=True,allow_soft_placement=0)
    config_proto.gpu_options.allow_growth = True
    sess=tf.Session(config=config_proto)
    sess.run(tf.global_variables_initializer())
    model.set_Session(sess)
    
    return model

dtypes = {
        'MachineIdentifier':                                    'category',
        'ProductName':                                          'category',
        'EngineVersion':                                        'category',
        'AppVersion':                                           'category',
        'AvSigVersion':                                         'category',
        'IsBeta':                                               'int8',
        'RtpStateBitfield':                                     'float16',
        'IsSxsPassiveMode':                                     'int8',
        'DefaultBrowsersIdentifier':                            'float16',
        'AVProductStatesIdentifier':                            'float32',
        'AVProductsInstalled':                                  'float16',
        'AVProductsEnabled':                                    'float16',
        'HasTpm':                                               'int8',
        'CountryIdentifier':                                    'int16',
        'CityIdentifier':                                       'float32',
        'OrganizationIdentifier':                               'float16',
        'GeoNameIdentifier':                                    'float16',
        'LocaleEnglishNameIdentifier':                          'int8',
        'Platform':                                             'category',
        'Processor':                                            'category',
        'OsVer':                                                'category',
        'OsBuild':                                              'int16',
        'OsSuite':                                              'int16',
        'OsPlatformSubRelease':                                 'category',
        'OsBuildLab':                                           'category',
        'SkuEdition':                                           'category',
        'IsProtected':                                          'float16',
        'AutoSampleOptIn':                                      'int8',
        'PuaMode':                                              'category',
        'SMode':                                                'float16',
        'IeVerIdentifier':                                      'float16',
        'SmartScreen':                                          'category',
        'Firewall':                                             'float16',
        'UacLuaenable':                                         'float32',
        'Census_MDC2FormFactor':                                'category',
        'Census_DeviceFamily':                                  'category',
        'Census_OEMNameIdentifier':                             'float16',
        'Census_OEMModelIdentifier':                            'float32',
        'Census_ProcessorCoreCount':                            'float16',
        'Census_ProcessorManufacturerIdentifier':               'float16',
        'Census_ProcessorModelIdentifier':                      'float16',
        'Census_ProcessorClass':                                'category',
        'Census_PrimaryDiskTotalCapacity':                      'float32',
        'Census_PrimaryDiskTypeName':                           'category',
        'Census_SystemVolumeTotalCapacity':                     'float32',
        'Census_HasOpticalDiskDrive':                           'int8',
        'Census_TotalPhysicalRAM':                              'float32',
        'Census_ChassisTypeName':                               'category',
        'Census_InternalPrimaryDiagonalDisplaySizeInInches':    'float16',
        'Census_InternalPrimaryDisplayResolutionHorizontal':    'float16',
        'Census_InternalPrimaryDisplayResolutionVertical':      'float16',
        'Census_PowerPlatformRoleName':                         'category',
        'Census_InternalBatteryType':                           'category',
        'Census_InternalBatteryNumberOfCharges':                'float32',
        'Census_OSVersion':                                     'category',
        'Census_OSArchitecture':                                'category',
        'Census_OSBranch':                                      'category',
        'Census_OSBuildNumber':                                 'int16',
        'Census_OSBuildRevision':                               'int32',
        'Census_OSEdition':                                     'category',
        'Census_OSSkuName':                                     'category',
        'Census_OSInstallTypeName':                             'category',
        'Census_OSInstallLanguageIdentifier':                   'float16',
        'Census_OSUILocaleIdentifier':                          'int16',
        'Census_OSWUAutoUpdateOptionsName':                     'category',
        'Census_IsPortableOperatingSystem':                     'int8',
        'Census_GenuineStateName':                              'category',
        'Census_ActivationChannel':                             'category',
        'Census_IsFlightingInternal':                           'float16',
        'Census_IsFlightsDisabled':                             'float16',
        'Census_FlightRing':                                    'category',
        'Census_ThresholdOptIn':                                'float16',
        'Census_FirmwareManufacturerIdentifier':                'float16',
        'Census_FirmwareVersionIdentifier':                     'float32',
        'Census_IsSecureBootEnabled':                           'int8',
        'Census_IsWIMBootEnabled':                              'float16',
        'Census_IsVirtualDevice':                               'float16',
        'Census_IsTouchEnabled':                                'int8',
        'Census_IsPenCapable':                                  'int8',
        'Census_IsAlwaysOnAlwaysConnectedCapable':              'float16',
        'Wdft_IsGamer':                                         'float16',
        'Wdft_RegionIdentifier':                                'float16',
        'HasDetections':                                        'int8'
        }
print('Loading Train and Test Data.\n')



train = pd.read_csv('/cos_person/Microsoft/train.csv', dtype=dtypes, low_memory=True)
# train = train.sample(frac=0.2, random_state=666)
train['MachineIdentifier'] = train.index.astype('uint32')
test  = pd.read_csv('./cos_person/Microsoft/test.csv',  dtype=dtypes, low_memory=True)
# test = test.sample(frac=0.2, random_state=666)
test['MachineIdentifier']  = test.index.astype('uint32')
test['HasDetections']=[0]*len(test)

def make_bucket(data,num=10):
    data.sort()
    bins=[]
    for i in range(num):
        bins.append(data[int(len(data)*(i+1)//num)-1])
    return bins
float_features=['Census_SystemVolumeTotalCapacity','Census_PrimaryDiskTotalCapacity']
for f in float_features:
    train[f]=train[f].fillna(1e10)
    test[f]=test[f].fillna(1e10)
    data=list(train[f])+list(test[f])
    bins=make_bucket(data,num=50)
    train[f]=np.digitize(train[f],bins=bins)
    test[f]=np.digitize(test[f],bins=bins)
    
train, dev,_,_ = train_test_split(train,train['HasDetections'],test_size=0.02, random_state=2019)
features=train.columns.tolist()[1:-1]

hparam=tf.contrib.training.HParams(
            model='nffm',
            norm=True,
            batch_norm_decay=0.9,
            hidden_size=[128,128],
            k=16,#16
            hash_ids=int(2e5),
            batch_size=1024,
            optimizer="adam",
            learning_rate=0.001,#0.001
            num_display_steps=1000,
            num_eval_steps=1000,
            epoch=1,#1
            metric='auc',
            init_method='uniform',
            init_value=0.1,
            feature_nums=len(features),
            kfold=4)
print_hparams(hparam)

for i in range(hparam.kfold):
    print("Fold",i)
    train=train.sample(frac=1)
    model=build_model(hparam)
    model.train(train_data=(train[features],train['HasDetections']),\
                dev_data=(dev[features],dev['HasDetections']))
    print("Training Done! Inference...")
    if i==0:
        preds=model.infer(dev_data=(test[features],test['HasDetections']))/hparam.kfold
    else:
        preds+=model.infer(dev_data=(test[features],test['HasDetections']))/hparam.kfold

submission = pd.DataFrame()
submission['MachineIdentifier'] = test['MachineIdentifier']
submission['HasDetections'] = preds
print(submission['HasDetections'].head())
submission.to_csv('/cos_person/Microsoft/nffm_sub_20190301_1.csv', index=False)