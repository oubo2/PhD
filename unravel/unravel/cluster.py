from unravel import *

import numpy as np
import random
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import FeatureAgglomeration
from sklearn.cluster import DBSCAN
from sklearn.cluster import OPTICS
import matplotlib.pyplot as plt
import stopwords
import difflib
import Levenshtein

def histplot(data):
    plt.hist(data, bins=data.max())
    plt.show()

def d(s1, s2):
    return 1 - difflib.SequenceMatcher( None
                                      , s1.lower().split()
                                      , s2.lower().split() ).ratio()

def cull(data):
    cols = data.columns.to_list()
    # Remove relationship grid columns: 57
    cs1 = {col for col in cols if 'urx' in col}
    # 'Replicate weight' columns: 270
    cs2 = { col for col in cols
                if 'Replicate weight' in meta.column_names_to_labels[col] }
    # 'Imputation flag' columns: 125
    cs3 = { col for col in cols
                if 'Imputation flag' in meta.column_names_to_labels[col] }
    # 'Population weight' columns: 8
    cs4 = { col for col in cols
                if 'Population weight' in meta.column_names_to_labels[col] }
    # 'interview outcome' columns: 30
    cs5 = { col for col in cols
                if 'interview outcome' in meta.column_names_to_labels[col] }
    # '(relationship)' columns: 45
    cs6 = { col for col in cols
                if '(relationship)' in meta.column_names_to_labels[col] }
    # 'Relationship in household' columns: 10
    cs7 = { col for col in cols
                if 'Relationship in household'
                in meta.column_names_to_labels[col] }
    # 'Income unit' columns: 11
    cs8 = { col for col in cols
                if 'Income unit' in meta.column_names_to_labels[col] }
    # 'Family type' columns: 10
    cs9 = { col for col in cols
                if 'Family type' in meta.column_names_to_labels[col] }
    # 'Family number person' columns: 11
    cs10 = { col for col in cols
                if 'Family number person' in meta.column_names_to_labels[col] }
    # 'Relationship of self' columns: 13
    cs11 = { col for col in cols
                 if 'Relationship of self' in meta.column_names_to_labels[col] }
    # 'Enumerated person' columns: 4
    cs12 = { col for col in cols
                 if 'Enumerated person' in meta.column_names_to_labels[col] }
    # 'Imputed age' columns: 8
    cs13 = { col for col in cols
                 if 'Imputed age' in meta.column_names_to_labels[col] }
    # 'Wave last interviewed' columns: 8
    cs14 = { col for col in cols
                 if 'Wave last interviewed' in meta.column_names_to_labels[col] }
    tocull = set.union( cs1, cs2, cs3, cs4, cs5
                      , cs6, cs7, cs8, cs9, cs10
                      , cs11, cs12, cs13, cs14
                      )
    for col in tocull: cols.remove(col)
    return data[cols], len(tocull)

def cols_in_cluster( data # Data set (pandas dataframe)
                   , clustering # Clustering object.
                   , index # Index of cluster to find columns of.
                   ):
    return [ meta.column_names_to_labels[col]
             for col in data.columns[np.where( clustering.labels_ == index
                                             , True
                                             , False )] ]
def vars_in_cluster( data # Data set (pandas dataframe)
                   , clustering # Clustering object.
                   , index # Index of cluster to find columns of.
                   ):
    return [ col for col in data.columns[np.where( clustering.labels_ == index
                                                 , True
                                                 , False )] ]

def pad(s1, s2):
    """Pad shortest string with spaces, return adjusted pair."""
    # Equal length strings. Do nothing.
    if len(s1) == len(s2): return s1, s2
    # String `s1` shorter. Pad it.
    elif len(s1) < len(s2):
        d = len(s2) - len(s1)
        return s1 + d*' ', s2
    # String `s2` shorter. Pad it.
    elif len(s1) > len(s2):
        d = len(s1) - len(s2)
        return s1, s2 + d*' '

def text_in_cluster(data, clustering, index):
    words = ' '.join(cols_in_cluster(data, clustering, index)).split()
    return [ word.lower() for word in words if word.isalpha() ]

def keywords(text, n=10, returndict=False):
    excludedwords = stopwords.get_stopwords('english')
    excludedwords.append('-')
    uwords = sorted( { word.lower()
                       for word in text
                       if word.lower() not in excludedwords } )
    d = dict( sorted( [ (word, text.count(word)) for word in uwords ]
                    , key=lambda t: t[1]
                    , reverse=True ) )
    if returndict: return d
    else: return list(d.keys())[:n]

def clusterkeywords(data, clustering, index, n=10, returndict=False):
    return keywords( text_in_cluster(data, clustering, index)
                   , n=n, returndict=returndict )

def dmatrix(labels):
    A = np.zeros((len(labels), len(labels)))
    for i in range(len(labels)):
        for j in range(1+i, len(labels)):
            A[i, j] = A[j, i] = d(labels[i], labels[j])
    return A

def cluster(data):
    labels = np.array( [ meta.column_names_to_labels[col]
                         for col in data.columns ] )
    A = dmatrix(labels)
    clustering = OPTICS(metric='precomputed').fit(A)

    clustersize = len(labels)
    iteration = 1
    while clustersize > 399:
        # Find largest cluster.
        bins, contents = np.unique(clustering.labels_, return_counts=True)
        argmx = contents.argmax()
        clusterindex = bins[argmx]     # Index of biggest cluster.
        clustersize = contents[argmx]  # Size of biggest cluster.
        lowestindex = bins.min()       # Lowest bin index, e.g. -1.
        highestindex = bins.max()      # Highest bin index, e.g. 254.
        print( "Largest cluster is %i, containing %i variables."
             % (clusterindex, clustersize) )
        print( "Recursively clustering largest cluster. Iteration: %i"
             % iteration)
        iteration += 1
        B = dmatrix(labels[clustering.labels_ == clusterindex])
        clusteringB = OPTICS( min_samples=2
                            , metric='precomputed').fit(B)
        # If only one cluster is found, report and break loop.
        if np.unique(clusteringB.labels_).shape[0] == 1:
            print("Only one cluster found. Nothing to be done.")
            break
        else:
            # Outliers are different outliers than before, thus different bin.
            mask = np.where( clusteringB.labels_ == -1
                           , lowestindex - 1
                           , clusteringB.labels_ )
            # First bin (i.e. index zero) goes in previous bin, rest to end.
            mask = np.where( mask == 0
                           , clusterindex
                           , highestindex + mask )
            # Apply.
            clustering.labels_[clustering.labels_ == clusterindex] = mask
    return clustering

def sample_var_from_cluster(data, clustering, clusterindex):
    vars = vars_in_cluster(data, clustering, clusterindex)
    np.random.seed(9)
    return np.random.choice(vars)

def subset_from_clusters(data, clustering):
    vars = []
    indices = np.unique(clustering.labels_, return_counts=True)[0]
    for i in indices:
        vars.append(sample_var_from_cluster(data, clustering, i))
    return data[vars]

def keyphrases(stringlist):
    pairs = [ (s1.lower().split(), s2.lower().split()) for s1 in stringlist
                                                       for s2 in stringlist
                                                       if s1 != s2 ]
    matches = []
    for pair in pairs:
        # Distill the matching blocks of words.
        s = difflib.SequenceMatcher(None, *pair)
        blocks = s.get_matching_blocks()[:-1] # '-1' to exclude trivial match.
        matching = [ pair[0][block.a:block.a+block.size]
                     for block in blocks ]
        # Remove non-words.
        matching = [ [ word for word in match if word.isalpha() ]
                     for match in matching ]
        # Concatenate words to phrases.
        matching = [ " ".join(match) for match in matching ]
        # Remove empty phrases.
        matching = [ match for match in matching if match != '' ]
        # Avoid repeating phrases across match-sets.
        matching = [ match for match in matching if match not in matches ]
        # Add to the list.
        matches += matching
    return sorted(matches, key=len, reverse=True)

def keyphrase(stringlist):
    if len(keyphrases(stringlist)) > 4:
        return keyphrases(stringlist)[0]
    else:
        words = keywords(' '.join(stringlist).split())
        words = [ word for word in words if word.isalpha() ]
        return ' '.join(words)

def discover_clustered( clustering
                      , algolist
                      , data
                      , chunksize=None
                      , target=None
                      , seed = None ):
    # Initialise the randomness.
    if seed is None:
        random.seed()
    else:
        random.seed(seed)
    # Get labels and size of clusters.
    indices, contents = np.unique(clustering.labels_, return_counts=True)
    # Sample one variable per cluster and collect in dictionary with keyphrase.
    variables = { random.choice(vars_in_cluster(data, clustering, index)):
                  keyphrase(cols_in_cluster(data, clustering, index))
                  for index in indices }
    # Do causal discovery on `clustered' data, i.e. on subsetted data.
    g = discover(algolist, data[variables.keys()], chunksize, target)
    # Deliver.
    return g, variables


if __name__ == "__main__":
    h, _ = cull(hilda)
    h33, _ = cull(hilda_by_isco(33))









