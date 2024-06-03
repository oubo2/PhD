# TODO: use AIC and BIC, maybe limit to max 2 outedge
'''
Calculate 
# p, q: model order/ number of lags, T: total number of observations
# F(p, T - p - q - 1) at 5%, F(4, 250000 - 4 - 4 - 1)
threshold = f.ppf(1 - 0.05, max_lag, 250000)
print(threshold)

F_statistics = result[max_lag][0]['ssr_ftest'][0]
print(F_statistics)
'''

import pandas as pd
import numpy as np
import networkx as nx
from networkx.drawing.nx_agraph import write_dot, graphviz_layout
from statsmodels.tsa.stattools import grangercausalitytests
from scipy.stats import f
# this does not work with continuous variable directly
from pyinform.transferentropy import transfer_entropy
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def grangerCausalityTable(df, lag = 4, threshold = 0.05):
    kFeatures = len(df.columns)
    # if prob < 0.05, it rejects the null hypothesis and indicate the variable in the row has influence has influence
    # on the variable in column
    GCTable = [[0 for i in range(kFeatures)] for j in range(kFeatures)]
    for x in range(kFeatures):
        for y in range(kFeatures):
            result = grangercausalitytests(df.iloc[:, [x,y]], lag, verbose=False)
            probability = result[lag][0]['ssr_ftest'][1]
            GCTable[x][y] = round(probability, 3)
            if GCTable[x][y] < threshold:
                GCTable[x][y] = True
            else:
                GCTable[x][y] = False
    return GCTable

# x -> x; x, y -> x
def transferEntropyTable(df, lag = 4, threshold = 0.1):
    kFeatures = len(df.columns)
    TETable = [[0 for i in range(kFeatures)] for j in range(kFeatures)]
    for x in range(kFeatures):
        for y in range(kFeatures):
            TETable[x][y] = transfer_entropy(df.iloc[:,x], df.iloc[:, y], k = lag)
            if TETable[x][y] > threshold:
                TETable[x][y] = True
            else:
                TETable[x][y] = False
    return TETable

# could use product from itertools
def generate_combinations(arrays):
    def helper(arrays, index, current, result):
        if index == len(arrays):
            result.append(current.copy())
            return
        for item in arrays[index]:
            current.append(item)
            helper(arrays, index + 1, current, result)
            current.pop()

    result = []
    helper(arrays, 0, [], result)
    return result

# could use product from itertools
def generate_structured_combinations(arrays):
    def helper(arrays, index, current, result, groups):
        if index == len(arrays):
            result.append(current.copy())
            return
        for item in arrays[index]:
            current.append(item)
            if index == 0:
                new_group = []
                groups.append(new_group)
            helper(arrays, index + 1, current, groups[-1], groups)
            current.pop()

    groups = []
    helper(arrays, 0, [], [], groups)
    return groups
    
def getTableEdges(table, feature_list):
    kFeatures = len(feature_list)
    edges = []
    for i in range(kFeatures):
        for j in range(kFeatures):
            if table[i][j]:
                edges.append((feature_list[i],feature_list[j]))
    return edges

def getNodesFromEdges(edges):
    nodes = []
    for i in edges:
        for j in i:
            if j not in nodes:
                nodes.append(j)
    return nodes

def alarm_df(df_normal, df_faulty):
    # rare event alarm is set to trigger when process variables go beyond 6 standard deviations from the mean value measured
    # at normal operating conditions
    normal_mean = df_normal.mean()
    normal_std = df_normal.std()
    feature_list = list(df_normal.columns)
    lower = normal_mean - normal_std * 6
    upper = normal_mean + normal_std * 6
    df_alarm = pd.DataFrame(0, index = np.arange(df_faulty.shape[0]), columns=feature_list)
    for feature in feature_list:
        alarm = []
        for i in df_faulty[feature]:
            if i <= lower[feature] or i >= upper[feature]:
                alarm.append(1)
            else:
                alarm.append(0)
        if sum(alarm) == 0:
            # not index 0, need to be greater than kLag
            alarm[10] = 1
        df_alarm[feature] = alarm
    '''
    for feature in feature_list:
        alarm = []
        for i in df_faulty[feature]:
            if i <= lower[feature]:
                alarm.append(2)
            elif i >= upper[feature]:
                alarm.append(1)
            else:
                alarm.append(0)
        if sum(alarm) == 0:
            # not index 0, need to be greater than kLag
            alarm[10] = 1
        df_alarm[feature] = alarm
    '''
    return df_alarm

# Function to calculate the probability of a single variable
def calculate_probability(df, var, value):
    total_count = len(df)
    var_count = len(df[df[var] == value])
    return var_count / total_count

# Function to calculate the conditional probability P(X|Y)
def calculate_conditional_probability(df, var_X, value_X, parents, parents_values):
    if len(parents) == 1:
        df = df[df[parents] == parents_values]
    else:
        for i in range(len(parents)):
            df = df[df[parents[i]] == parents_values[i]]
    total_count = len(df)
    if total_count == 0:
        return 0
    joint_count = len(df[df[var_X] == value_X])
    return joint_count / total_count
  
def drawBN(edges):
    G=nx.DiGraph()
    nodes = getNodesFromEdges(edges)
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    pos =graphviz_layout(G, prog='dot')
    nx.draw(G, pos, with_labels=True, arrows=True)
    plt.show()
'''
    nx.draw_circular(
        G, with_labels=True, arrowsize=30, node_size=800, alpha=0.3, font_weight="bold"
    )
    plt.show()'''

def getPCANumComponents(pca):
    explained_variance = 100*pca.explained_variance_ratio_ # in percentage
    cum_explained_variance = np.cumsum(explained_variance) # cumulative % variance explained

    n_comp = np.argmax(cum_explained_variance >= 95) + 1
    return n_comp

# variable contribution
from tsquared import HotellingT2, MYTDecomposition
from xgboost import XGBRegressor
#xgboost DCIG
def featureDCIG(df_normal, df_faulty, verbose=False):
    # df_normal and df_faulty should be standardscaled
    hotelling = HotellingT2().fit(df_normal)
    normal_t2_score = hotelling.score_samples(df_normal)
    faulty_t2_score = hotelling.score_samples(df_faulty)

    model = XGBRegressor()
    model.fit(df_normal, normal_t2_score)
    normalFeatureImportance = model.feature_importances_
    model.fit(df_faulty, faulty_t2_score)
    faultyFeatureImportance = model.feature_importances_
    diff = abs(faultyFeatureImportance - normalFeatureImportance) / normalFeatureImportance
    DCIG = diff / sum(diff)
    if verbose:
        kFeatures = len(df_normal.columns)
        features = list(df_normal.columns)
        print([i for i in range(kFeatures) if DCIG[i] > 1/kFeatures])
        plt.bar(features, faultyFeatureImportance)
        plt.xticks(rotation='vertical')
        plt.show()

    return DCIG

def QContributions(df_normal, df_faulty, verbose=False):
    features = list(df_normal.columns)

    scaler = StandardScaler()
    scaler.fit(df_normal)
    df_normal = scaler.transform(df_normal)
    df_faulty = scaler.transform(df_faulty)
    
    pca = PCA()
    score_train = pca.fit_transform(df_normal)
    k = getPCANumComponents(pca)
    V_matrix = pca.components_.T
    P_matrix = V_matrix[:,:k]
    lambda_k = np.diag(pca.explained_variance_[0:k]) # eigenvalue = explained variance
    lambda_k_inv = np.linalg.inv(lambda_k)
    
    # compute scores and reconstruct
    score_train_reduced = score_train[:,0:k]
    data_train_normal_reconstruct = np.dot(score_train_reduced, P_matrix.T)
    score_test = pca.transform(df_faulty)
    score_test_reduced = score_test[:,0:k]

    data_test_normal_reconstruct = np.dot(score_test_reduced, P_matrix.T)

    error_train = df_normal - data_train_normal_reconstruct
    Q_train = np.sum(error_train*error_train, axis = 1)
    Q_CL = np.percentile(Q_train, 99)

    error_test = data_test_normal_reconstruct - df_faulty
    Q_test = np.sum(error_test*error_test, axis = 1)

    # plot Q_test and Q_train with CL
    if verbose:
        plt.figure(figsize=[6,4])
        plt.plot(Q_test, color='black')
        plt.plot([1,len(Q_test)],[Q_CL,Q_CL], linestyle='--',color='red', linewidth=2)
        plt.xlabel('Sample #')
        plt.ylabel('Q metric: faulty data')
        plt.show()

        sample = 250
        error_test_sample = error_test[sample-1,]
        Q_contri = error_test_sample*error_test_sample # vector of contributions
        plt.figure(figsize=[15,4])
        plt.bar(features, Q_contri)
        plt.xticks(rotation = 80)
        plt.ylabel('Q contributions')
        plt.show()
    
    # get average contribution of variables for all samples over control limit
    Q_contris = np.zeros(len(features))
    for error_sample in error_test:
        contri = error_sample*error_sample
        if sum(contri) > Q_CL:
            Q_contris = np.add(Q_contris, contri) 

    return Q_contris