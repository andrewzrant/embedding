import os
import time
import pandas as pd
import copy
os.system('pip install node2vec')
#os.system('pip install xgboost -gpu')
# os.system('pip install tqdm')
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


import pandas as pd
import numpy as np
import gc
from multiprocessing import Pool
import time
import os
from sklearn.preprocessing import LabelEncoder
import networkx as nx
from node2vec import Node2Vec
os.chdir("/cos_person/Elo/data" )

his_trans = pd.read_csv('../data/historical_transactions.csv')
his_trans = his_trans[his_trans.authorized_flag == 'Y'][[ 'card_id','merchant_id' ]]
cacheRoot = '../cache/'
def create_edgelist(useData, dataID ,secNodeCol):
    print('creating edge list ... ')
    le = LabelEncoder()
    datacp = useData[[dataID,secNodeCol ]]
    datacp = datacp[-datacp[secNodeCol].isnull()]
    datacp[secNodeCol] = le.fit_transform( datacp[secNodeCol] )+ 1000000
    print('card_id label encode ')
    fiter = le.fit_transform( datacp[ dataID ])
    print('card_id label encode k v s')
    idkeys = dict(zip(datacp[ dataID ],fiter ))
    datacp[ dataID ] = datacp[ dataID ].map( idkeys )
    del idkeys
    idvals = dict(zip(fiter,datacp[ dataID ] ))
    gp =datacp.groupby([dataID,secNodeCol])[dataID].agg({"purchase_user_mer":'count'}).reset_index()
    gpmer = gp.groupby( dataID )['purchase_user_mer'].sum()
    gp['purchase_user'] = gp[dataID].map( gpmer )
    gp[ "weight"] = gp['purchase_user_mer']/gp['purchase_user']
    gp = gp.drop(['purchase_user_mer','purchase_user'],axis = 1)
    savename = cacheRoot +  '{}_weighted_edglist.txt'.format(secNodeCol)
    np.savetxt(savename, gp.values, fmt=['%d','%d','%f'])
    del gp
    return savename,idvals


def createEdgeFomat(fname):
    G = nx.Graph()
    f = open(fname,'r')
    lines = f.readlines()
    f.close()
    lines =[l.replace("\n","").split(" ")  for l in lines]
    lines = [[int(x[0]),int(x[1]),float(x[2])] for x in lines]
    edfname = fname.replace(".txt",".edgelist")
    
    for edg in lines:
        G.add_edge(edg[0], edg[1], weight=edg[2])
    print("\n-------------------------------------\n")
    print("saving fali name %s " % edfname)
    print("\n-------------------------------------\n")
    fh=open(edfname,'wb')
    nx.write_edgelist(G, fh)
    fh.close()
    return edfname


def emb_graph_2vec(inputpath,dim):
    print("input name will be ",inputpath)
    emb_name = inputpath.replace("weighted_edglist.edgelist","")
    print("emb_name will be ",emb_name)

    savename =inputpath.replace("weighted_edglist.edgelist",".emb")
    print("emb outfile name will be ",savename)
    if os.path.exists(savename):
        print("file alread exists in cache, please rename")
        sys.exit(1)
    print('read graph ')
    graph = nx.read_edgelist(inputpath,create_using=nx.DiGraph())
    print('create node2vec model ')
    node2vec = Node2Vec(graph, dimensions=dim, walk_length=30, num_walks=200, workers=10 ) 
    # Embed nodes
    print("training .... ")
    model = node2vec.fit(window=10, min_count=1, batch_words=4)
    print("training finished saving result... ")

    print("saving %s file to disk "%savename)
    # Save embeddings for later use
    model.wv.save_word2vec_format(savename)
    print("done")
    return savename
def get_embeding(fname, embname):
    f = open(fname)
    embeding_lines = f.readlines()
    f.close()
    mapfunc = lambda x: list( map( float, x ) )
    embeding_lines = [li.replace("\n","").split(" ") for li in embeding_lines[1:]]
    embeding_lines = [ [ int(line[0]) ] +  mapfunc( line[1:]  )   for line in   embeding_lines ]
    cols = ["card_id"] + [ embname  + str(i) for i in range( len(embeding_lines[0]) -1 )]
    embeding_df = pd.DataFrame(embeding_lines, columns=cols )
    del embeding_lines
    return embeding_df
    
print('create edge list txt')
savename,idvals = create_edgelist(his_trans,'card_id','merchant_id' )
print('txt to edglist ')
edfname = createEdgeFomat(savename)
print('edglist fit to node2vec as graph ')
savename = emb_graph_2vec(edfname,36)
print('get embeddings ')
embeding_df = get_embeding(savename, 'n2v_merchants_')
embeding_df[ 'card_id' ] = embeding_df[ 'card_id' ].map( idvals )
embeding_df.to_csv('merchants_36dims_embedding.csv', index = False)