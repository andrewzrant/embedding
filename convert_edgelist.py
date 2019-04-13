import os
import time
import pandas as pd
import copy
# os.system('pip install deepwalk')
#os.system('pip install xgboost -gpu')
# os.system('pip install tqdm')
# from tqdm import *
# from node2vec import Node2Vec

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



def edge_listtxt(inputpath, outputpath):
    f = open(inputpath )
    lines = f.readlines()
    lines =  [ " ".join(c.split(" ")[:2]) for c in lines  ]
    f.close()
    linesall = "\n".join(lines)
    f = open(outputpath,"w" )
    f.write(linesall )
    f.close()

inputP = "/cos_person/fusai/"
output = "/cos_person/deepwalk_edges/"

files = os.listdir(inputP)
files = [ c for c in files if ".edgelist" in c]
for cn in files:
	inputpath = inputP + cn
	outputpath = output + cn.replace(".edgelist",".txt" )
	print("converting edgelist: %s into  txt file: %s ... " % (inputpath , outputpath ) )
	try:
		edge_listtxt(inputpath, outputpath)
		print("successfully converted")
	except Exception as e:
		print("convert failed , error :", str(e))