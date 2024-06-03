from util import *
from pysmileUtil import *

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, KernelPCA
from sklearn.preprocessing import StandardScaler
from tsquared import HotellingT2, MYTDecomposition
from xgboost import XGBRegressor

from pgmpy.estimators import PC, HillClimbSearch, ExhaustiveSearch, BicScore, TreeSearch, MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
from pgmpy.models import BayesianNetwork

import pyreadr
import os 
current_directory = os.path.dirname(os.path.realpath(__file__))
train_normal_path = current_directory + '\\TEP2017R\\TEP_FaultFree_Training.RData'
train_faulty_path =  current_directory + '\\TEP2017R\\TEP_Faulty_Training.RData'
 
train_normal = pyreadr.read_r(train_normal_path)['fault_free_training']
train_faulty = pyreadr.read_r(train_faulty_path)['faulty_training']

# only keep continuous variables, xmeas_1 - xmeas_22
df_train_normal = train_normal[train_normal.simulationRun==1].iloc[:,3:25]
df_train_faulty = train_faulty[(train_faulty.simulationRun==1) & (train_faulty.faultNumber==12)].iloc[:,3:25]

kFeatures = len(df_train_normal.columns)
features = list(df_train_normal.columns)
ss = StandardScaler().set_output(transform="pandas")
ss_df_train_normal = ss.fit_transform(df_train_normal)

#KPCA
kpca = KernelPCA(n_components=4, kernel='rbf', fit_inverse_transform=True)
x_faulty = ss.transform(df_train_faulty,copy=None)
x_kpca = kpca.fit(ss_df_train_normal)
x_fault_free_reconstructed = kpca.inverse_transform(kpca.transform(ss_df_train_normal))
x_faulty_reconstructed = kpca.inverse_transform(kpca.transform(x_faulty))

normal_residuals = x_fault_free_reconstructed - ss_df_train_normal
faulty_residuals = x_faulty_reconstructed - x_faulty

df_train_normal_residuals = pd.DataFrame(data=normal_residuals, columns=features)
df_train_faulty_residuals = pd.DataFrame(data=faulty_residuals, columns=features)
print(df_train_normal_residuals)

df_alarm = alarm_df(df_train_normal_residuals, df_train_faulty_residuals)
TETable = transferEntropyTable(df_alarm, 4, 0.08)
edges = getTableEdges(TETable, features)
nodes = getNodesFromEdges(edges)
net = getBayesianNet(df_alarm)
for node in net.get_all_nodes():
    print_node_info(net, node)