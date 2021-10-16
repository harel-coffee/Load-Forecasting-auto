#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.decomposition import TruncatedSVD
import random
import seaborn as sns
import os.path as path
import os
import matplotlib
import matplotlib.font_manager
import matplotlib.pyplot as plt # graphs plotting
from Bio import SeqIO # some BioPython that will come in handy
#matplotlib inline
import numpy
import csv 

from matplotlib import rc

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn import svm

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics

import pandas as pd
import sklearn
from sklearn import preprocessing
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler  
from sklearn.neural_network import MLPClassifier 
from sklearn.metrics import classification_report, confusion_matrix 

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

from pandas import DataFrame

from sklearn.model_selection import KFold 
from sklearn.model_selection import RepeatedKFold

from sklearn.metrics import confusion_matrix

from numpy import mean


from itertools import cycle

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from sklearn.metrics import roc_auc_score

import statistics

from sklearn.cluster import KMeans

from sklearn.datasets import load_digits
from sklearn.decomposition import KernelPCA

import math
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import confusion_matrix

# for Arial typefont
matplotlib.rcParams['font.family'] = 'Arial'


## for Palatino and other serif fonts use:
# rc('font',**{'family':'serif','serif':['Palatino']})
# matplotlib.rcParams['mathtext.fontset'] = 'cm'

## for LaTeX typefont
# matplotlib.rcParams['mathtext.fontset'] = 'stix'
# matplotlib.rcParams['font.family'] = 'STIXGeneral'

## for another LaTeX typefont
# rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})

# rc('text', usetex = True)

print("Packages imported")


# In[7]:


path_tmp = "E:/RA/Load Forecasting (Smart City)/Dataset/"
read_path = path_tmp + "Australia_60_Min_Data_26304_x_36_Original.csv"

variants_names = []

with open(read_path) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        tmp = row
#         aa_1 = str(tmp).replace("[","")
#         aa_2 = str(aa_1).replace("]","")
#         aa_3 = str(aa_2).replace("\'","")
        variants_names.append(tmp)


# In[8]:


len(variants_names),len(variants_names[0])


# In[13]:


final_load = []

for i in range(len(variants_names)):
    tmp_new = variants_names[i]
    tmp_list = []
    for j in range(len(tmp_new)):
        tmp_list.append(float(tmp_new[j]))
    final_load.append(tmp_list)
    
    
    


# In[14]:


len(final_load),len(final_load[0])


# In[18]:


from sklearn.manifold import TSNE

# We embed all our sequences into 2D vectors with help of TSNE
X_embedded = TSNE(n_components = 2, perplexity = 30, random_state = 1).fit_transform(final_load)


# In[ ]:


np.save("australia_t_sne_data.npy",X_embedded)

