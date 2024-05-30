from util import *
import pysmileUtil
import matplotlib.pyplot as plt
#import networkx as nx
#from networkx.drawing.nx_agraph import write_dot, graphviz_layout
import numpy as np

import pandas as pd
from sklearn.decomposition import PCA, KernelPCA
from sklearn.preprocessing import StandardScaler
#information gain
#from sklearn.feature_selection import mutual_info_classif

# variable contribution
from tsquared import HotellingT2, MYTDecomposition
from xgboost import XGBRegressor


# Bayesian
from pgmpy.estimators import PC, HillClimbSearch, ExhaustiveSearch, BicScore, TreeSearch, MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
from pgmpy.models import BayesianNetwork

import pyreadr
import os 
current_directory = os.path.dirname(os.path.realpath(__file__))
train_normal_path = current_directory + '\\TEP2017R\\TEP_FaultFree_Training.RData'
train_faulty_path =  current_directory + '\\TEP2017R\\TEP_Faulty_Training.RData'
 
#test_normal_path = current_directory + '\\TEP2017R\\TEP_FaultFree_Testing.RData'
#test_faulty_path = current_directory + '\\TEP2017R\\TEP_Faulty_Testing.RData'
 
train_normal = pyreadr.read_r(train_normal_path)['fault_free_training']
train_faulty = pyreadr.read_r(train_faulty_path)['faulty_training']
#test_normal = pyreadr.read_r(test_normal_path)['fault_free_testing']
#test_faulty = pyreadr.read_r(test_faulty_path)['faulty_testing']

# only keep continuous variables, xmeas_1 - xmeas_22
df_train_normal = train_normal[train_normal.simulationRun==1].iloc[:,3:25]
df_train_faulty = train_faulty[(train_faulty.simulationRun==1) & (train_faulty.faultNumber==12)].iloc[:,3:25]
#df_test_normal = test_normal[test_normal.simulationRun==1].iloc[:,3:25]
kFeatures = len(df_train_normal.columns)
feature_list = list(df_train_normal.columns)

ss = StandardScaler().set_output(transform="pandas")
ss_df_train_normal = ss.fit_transform(df_train_normal)

# TEP truths
faultToRoot = {1:'xmeas_4', 4:'xmeas_9', 5:'xmeas_22', 6:'xmeas_1', 11:'xmeas_21', 12:'xmeas_11', 14:'xmeas_9', 15:'xmeas_11'}
faultToVariables = {1:['xmeas_1', 'xmeas_4', 'xmeas_18', 'xmeas_21', 'xmeas_25', 'xmeas_26'],
                    14:['xmeas_9', 'xmeas_11', 'xmeas_21', 'xmeas_32']}
reactor = ['xmeas_1', 'xmeas_2', 'xmeas_3', 'xmeas_4', 'xmeas_5', 'xmeas_6', 'xmeas_7', 'xmeas_8', 'xmeas_9', 
           'xmeas_20','xmeas_21']
seperator = ['xmeas_7', 'xmeas_20', 'xmeas_21', 'xmeas_10', 'xmeas_11', 'xmeas_12', 'xmeas_13', 'xmeas_14', 'xmeas_22']
stripper = ['xmeas_11', 'xmeas_12', 'xmeas_13', 'xmeas_14', 'xmeas_22', 'xmeas_15', 'xmeas_16', 'xmeas_17', 'xmeas_18', 
           'xmeas_19', 'xmeas_4', 'xmeas_5']

df_train_faulty_alarm = alarm_df(df_train_normal, df_train_faulty)
BN = pysmileUtil.getBayesianNet(df_train_faulty_alarm)
for node in BN.get_all_nodes():
    print(BN.get_node_id(node))

