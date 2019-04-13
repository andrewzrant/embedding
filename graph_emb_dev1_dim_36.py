import os
import time
import pandas as pd
import copy
os.system('pip install node2vec')
#os.system('pip install xgboost -gpu')
# os.system('pip install tqdm')
# from tqdm import *
from node2vec import Node2Vec

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

import networkx as nx

def emb_graph_2vec(inputpath,dim,emb_out):
	print("input name will be ",inputpath)
	emb_name = inputpath.split(".")[-2].split("/")[-1].replace("edgelist","")
	print("emb_name will be ",emb_name)

	savename = emb_out + emb_name + "_emb_dim"  + str(dim) + ".emb"
	print("emb outfile name will be ",savename)
	
	graph = nx.read_edgelist(inputpath,create_using=nx.DiGraph())
	# Precompute probabilities and generate walks - **ON WINDOWS ONLY WORKS WITH workers=1**
	node2vec = Node2Vec(graph, dimensions=dim, walk_length=30, num_walks=200, workers=10) 
	# Embed nodes
	print("training .... ")
	model = node2vec.fit(window=10, min_count=1, batch_words=4)
	print("training finished saving result... ")

	print("saving %s file to disk "%savename)
	# Save embeddings for later use
	model.wv.save_word2vec_format(savename)
	print("done")
	# Save model for later use
	
inputP = "/cos_person/fusai/"

inputpath = inputP + "device1_edglist.edgelist"
emb_out = "/cos_person/emb_out/"
try:
    emb_graph_2vec(inputpath,36,emb_out)
except Exception as e:
    print(e)