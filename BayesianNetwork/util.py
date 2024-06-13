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

# x -> x; x, y -> x
def scoreTable(df, scoreFunction, lag = 4, verbose=False):
    #sampling frequency
    #df = df.iloc[::5, :]
    kFeatures = len(df.columns)
    TETable = [[0 for i in range(kFeatures)] for j in range(kFeatures)]
    for x in range(kFeatures):
        if verbose:
            print(str(x/kFeatures*100) + "%")
        for y in range(kFeatures):
            # no need to hard lock function signature and pass lambda instead
            TETable[x][y] = scoreFunction(df, x, y, lag)
    return TETable

def transferEntropyScore(df, x, y, lag):
    #print(transfer_entropy(df.iloc[:,x], df.iloc[:, y], k = lag))
    #print( transfer_entropy(df.iloc[:,y], df.iloc[:, x], k = lag))
    return transfer_entropy(df.iloc[:,x], df.iloc[:, y], k = lag) - transfer_entropy(df.iloc[:,y], df.iloc[:, x], k = lag)

def transferEntropyContinuousScore(df, x, y, lag):
    return transfer_entropy_continuous(df.iloc[:,x], df.iloc[:, y], delay = lag) - transfer_entropy_continuous(df.iloc[:,y], df.iloc[:, x], delay= lag)

def grangerCausalityScore(df, x, y, lag):
    result = grangercausalitytests(df.iloc[:, [x,y]], lag, verbose=False)
    probability = round(result[lag][0]['ssr_ftest'][1], 3)
    return probability

def probabilityToBool(probabilityTable, threshold, greater=True):
    #gc less than
    #te greater than
    row, col = len(probabilityTable), len(probabilityTable[0])
    table = [[0] * col for _ in range(row)] 
    for r in range(row):
        for c in range(col):
            if probabilityTable[r][c] > threshold:
                table[r][c] = True
            else:
                table[r][c] = False
    if not greater:
        table = [[not j for j in i] for i in table]
    
    return table

def removeCycles(edges, score_Table, df):
    G=nx.DiGraph()
    nodes = getNodesFromEdges(edges)
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    cycles = list(nx.simple_cycles(G))
    removedEdges = []
    result = []
    # could remove dependency on df
    # this might remove more than one edge from a cycle, but whatever for now
    # a dictionary of edges to cycle index, and edges corresponding to cycle index
    features = list(df.columns)
    featureToIndex = {}
    for i in range(len(features)):
        featureToIndex[features[i]] = i

    for cycle in cycles:
        edges = [(cycle[i], cycle[i+1]) for i in range(len(cycle) - 1)] + [(cycle[-1],cycle[0])]
        common_edges = set(edges) & set(removedEdges)
        if common_edges:
            for i in common_edges:
                edges.remove(i)
            result.append(edges)
        else:
            edges_scores = [score_Table[featureToIndex[i]][featureToIndex[j]] for (i, j) in edges]
            weakest_edge_index = np.argmin(edges_scores)
            removedEdges.append(edges[weakest_edge_index])
            result.append(edges[:weakest_edge_index] + edges[weakest_edge_index+1:])

    return removedEdges

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
    
def getTableEdges(table, df):
    feature_list = list(df.columns)
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

import numpy as np
from scipy import stats
from scipy import ndimage



def transfer_entropy_continuous(X,Y,delay=1,gaussian_sigma=None):
	'''
	TE implementation: asymmetric statistic measuring the reduction in uncertainty
	for a future value of X given the history of X and Y. Or the amount
	of information from Y to X. Calculated through the Kullback-Leibler divergence 
	with conditional probabilities

	author: Sebastiano Bontorin
	mail: sbontorin@fbk.eu

	args:
		- X (1D array):
			time series of scalars (1D array)
		- Y (1D array):
			time series of scalars (1D array)
	kwargs:
		- delay (int): 
			step in tuple (x_n, y_n, x_(n - delay))
		- gaussian_sigma (int):
			sigma to be used
			default set at None: no gaussian filtering applied
	returns:
		- TE (float):
			transfer entropy between X and Y given the history of X
	'''

	if len(X)!=len(Y):
		raise ValueError('time series entries need to have same length')

	n = float(len(X[delay:]))

	# number of bins for X and Y using Freeman-Diaconis rule
	# histograms built with numpy.histogramdd
	binX = int( (max(X)-min(X))
				/ (2* stats.iqr(X) / (len(X)**(1.0/3))) )
	binY = int( (max(Y)-min(Y))
				/ (2* stats.iqr(Y) / (len(Y)**(1.0/3))) )

	# Definition of arrays of shape (D,N) to be transposed in histogramdd()
	x3 = np.array([X[delay:],Y[:-delay],X[:-delay]])
	x2 = np.array([X[delay:],Y[:-delay]])
	x2_delay = np.array([X[delay:],X[:-delay]])

	p3,bin_p3 = np.histogramdd(
		sample = x3.T,
		bins = [binX,binY,binX])

	p2,bin_p2 = np.histogramdd(
		sample = x2.T,
		bins=[binX,binY])

	p2delay,bin_p2delay = np.histogramdd(
		sample = x2_delay.T,
		bins=[binX,binX])

	p1,bin_p1 = np.histogramdd(
		sample = np.array(X[delay:]),
		bins=binX)

	# Hists normalized to obtain densities
	p1 = p1/n
	p2 = p2/n
	p2delay = p2delay/n
	p3 = p3/n

	# If True apply gaussian filters at given sigma to the distributions
	if gaussian_sigma is not None:
		s = gaussian_sigma
		p1 = ndimage.gaussian_filter(p1, sigma=s)
		p2 = ndimage.gaussian_filter(p2, sigma=s)
		p2delay = ndimage.gaussian_filter(p2delay, sigma=s)
		p3 = ndimage.gaussian_filter(p3, sigma=s)

	# Ranges of values in time series
	Xrange = bin_p3[0][:-1]
	Yrange = bin_p3[1][:-1]
	X2range = bin_p3[2][:-1]

	# Calculating elements in TE summation
	elements = []
	for i in range(len(Xrange)):
		px = p1[i]
		for j in range(len(Yrange)):
			pxy = p2[i][j]

			for k in range(len(X2range)):
				pxx2 = p2delay[i][k]
				pxyx2 = p3[i][j][k]

				arg1 = float(pxy*pxx2)
				arg2 = float(pxyx2*px)

				# Corrections avoding log(0)
				if arg1 == 0.0: arg1 = float(1e-8)
				if arg2 == 0.0: arg2 = float(1e-8)

				term = pxyx2*np.log2(arg2) - pxyx2*np.log2(arg1) 
				elements.append(term)

	# Transfer Entropy
	TE = np.sum(elements)
	return TE