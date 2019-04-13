import os
import time
import pandas as pd
import copy

os.system('pip install gensim')
os.system('pip install tqdm')
# from tqdm import *


from sklearn.model_selection import train_test_split, GridSearchCV,StratifiedKFold
import numpy as np
from sklearn.metrics import f1_score
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

def aggsW2V(x):
    x = x.astype(str)
    return ' '.join(x)

os.chdir("/cos_person/Elo/data" )
his_merchants = pd.read_csv('../data/historical_transactions.csv')
his_merchants['purchase_date'] = pd.to_datetime(his_merchants['purchase_date'])
his_merchants = his_merchants.sort_values('purchase_date',ascending=0)
his_merchants = his_merchants[ his_merchants.card_id.isnull() == False ]
# his_merchants['dayofweek'] = his_merchants['purchase_date'].dt.dayofweek
# his_merchants['hour'] = his_merchants['purchase_date'].dt.hour

w2vDF = pd.DataFrame()
w2vDF['card_id'] = his_merchants.card_id.unique().tolist()


from gensim.models import Word2Vec

for col in ['state_id','city_id','installments' ,'subsector_id','merchant_category_id','month_lag','purchase_amount' ]:
    print('training {} w2v features '.format( col ))
    print('step 1: grouping ... ')
    gp = his_merchants.groupby('card_id')[col].unique().reset_index()
    print('step 2: get sentence ... ')
    sentence = [ str( vl ) for line in gp[col].values.tolist() for vl in line  ]
    print('step 2:training sentence...',col)
    model = Word2Vec(sentence, size=8, window=8, min_count=1,iter=10, workers=10)
    print('step 3: get sentence w2v ... ')
    outdf = []
    for line in sentence:
        sumarr = 0
        for vcab in line:
            sumarr = sumarr + model[vcab]
        outdf.append(sumarr/len(line))
    tmp = pd.DataFrame(outdf)
    tmp.columns = [col + '_w2v_' +str(i) for i in tmp.columns]
    tmp['card_id'] = gp['card_id']
    w2vDF = w2vDF.merge(tmp , on = ['card_id'], how = 'left'  )
    del tmp,gp,sentence,outdf




    
w2vDF.to_csv('../cache/w2v_each_of.csv',index =False)