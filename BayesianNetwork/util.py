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
    return df_alarm

# Function to calculate the probability of a single variable
def calculate_probability(df, var, value):
    total_count = len(df)
    var_count = len(df[df[var] == value])
    return var_count / total_count
        
def drawBN(edges):
    G=nx.DiGraph()
    nodes = getNodesFromEdges(edges)
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    pos =graphviz_layout(G, prog='dot')
    nx.draw(G, pos, with_labels=True, arrows=True)
    plt.show()

    nx.draw_circular(
        G, with_labels=True, arrowsize=30, node_size=800, alpha=0.3, font_weight="bold"
    )
    plt.show()