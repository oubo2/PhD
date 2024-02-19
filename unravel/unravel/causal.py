#!/bin/env python3
#
# causal.py - causal modelling with HILDA data.
#
# Author: Fjalar de Haan (fjalar.dehaan@unimelb.edu.au)
# Created: 2023-02-14
# Last modified: 2023-05-21
#

import pickle, os, multiprocessing, copy, random

import cdt
from cdt.metrics import SHD, SID

cdt.SETTINGS.rpath = 'C:/Program Files/R/R-4.3.2/bin/Rscript'

from pyCausalFS.CBD.MBs.HITON.HITON_MB import HITON_MB

import networkx as nx
import pandas as pd
import numpy as np

from .gtools import markov_blanket

glasso = cdt.independence.graph.Glasso()

# Make sure SID returns an integer.
def SID(target, prediction): return int(cdt.metrics.SID(target, prediction))

# Instantiate pairwise algorithm.
anm = cdt.causality.pairwise.ANM()

# Instantiate all graph-based algorithms.
algorithms = [ cdt.causality.graph.CAM()
             , cdt.causality.graph.CCDr()
             , cdt.causality.graph.GES()
             , cdt.causality.graph.GIES()
             , cdt.causality.graph.LiNGAM()
             , cdt.causality.graph.PC()
             , cdt.causality.graph.SAM()
             , cdt.causality.graph.SAMv1() ]
algos = { str(algo).split(sep='.')[3]: algo for algo in algorithms }
nalgos = len(algos)

def partrand(cs, n, var=None):
    """Return random partition of `cs` as list of lists of size `n`."""
    cs = list(cs)
    random.shuffle(cs)
    partition = [ cs[i:i+n] for i in range(0, len(cs), n) ]
    # If the last list is too small, merge it with the one before.
    if len(partition[-1]) < n // 2:
        merged = partition[-2] + partition[-1]
        partition = partition[:-2]
        partition.append(merged)
    # If only a partition is wanted, all is done.
    if var is None:
        return partition
    # Otherwise make sure all parts contain `var`.
    else:
        for part in partition:
            if var not in part:
                part.append(var)
        return partition

def blanketsbychunks(data, algo='GIES', target='ujbmsall', chunksize=100):
    # First partition the set of variables into lists of size `chunksize`.
    partition = partrand(data.columns, chunksize, target)
    # Return the list of Markov blankets for each chunk of `data`.
    blankets = []
    for i, part in enumerate(partition):
        print("Computing blanket number %i." % i)
        blankets.append(markov_blanket(algos[algo].predict(data[part]), target))
    # return [ markov_blanket(algos[algo].predict(data[part]), target )
             # for part in partition ]
    return blankets

def candidates(variables, data, algo=anm, threshold=.1):
    """Return candidate causes/effects for `variables` in `data`."""
    # In case of single variable, put it in a list anyway.
    if type(variables) != list:
        variables = [variables]
    # Start with nothing.
    candidates = []
    # For each variable, do the pairwise comparison with each column.
    for variable in variables:
        score = 0 # Start afresh with each new variable.
        for i in range(data.shape[1]):
            score = algo.predict_proba((data[variable], data.iloc[:, i]))
            print(i, score)
            found = abs(score) < 1.5 and abs(score) > threshold
            if found:                              # If a candidate is found...
                candidates.append(data.columns[i]) # ... add it to the list.
    # Remove duplicates.
    candidates = list(set(candidates))
    # Deliver.
    return candidates

def distances(data, algos):
    algonames = list(algos.keys())
    nalgos = len(algonames)
    df = pd.DataFrame(index=algonames, columns=algonames)
    gs = [ algos[algo].predict(data) for algo in algos ]
    shd_matrix = df.copy()
    sid_matrix = df.copy()
    for row in range(nalgos):
        for col in range(nalgos):
            shd_matrix.loc[algonames[row], algonames[col]]=SHD(gs[row], gs[col])
            sid_matrix.loc[algonames[row], algonames[col]]=SID(gs[row], gs[col])
    return shd_matrix, sid_matrix

def blanket(data, var, algorithm=HITON_MB, alpha=.01):
    """Extract Markov blanket incl. seed var. as label list."""
    # Extract column names.
    cols = data.columns
    # Get index of variable `var`.
    index = cols.to_list().index(var)
    # Obtain Markov blanket of `var`.
    b = algorithm(data, index, alpha)[0]
    # Replace indices with labels.
    variables = list(cols[b])
    # Add source variable and return blanket as list of variable labels.
    return [var] + variables

def blankets(data, variables, algorithm=HITON_MB, alpha=.01, parallel=False):
    """Extract Markov blankets incl. seeds of each variable."""
    if parallel:
        pool = multiprocessing.Pool()
        blankets = {}
        for variable in variables:
            args = data, variable, algorithm, alpha
            blankets[variable] = pool.apply_async(blanket, args)
        blankets = { key: val.get() for (key, val) in blankets.items() }
    else:
        blankets = { variable: blanket(data, variable, algorithm, alpha)
                     for variable in variables }
    return blankets

def causal_blanket(data, variables, algo='GES', alpha=.01):
    """Do causal discovery on Markov blanket of `var`."""
    # If `var` is a list, assume it is a list of seed variables.
    if type(variables) == list:
        b = []
        for var in variables:
            b += blanket(data, var, alpha=alpha)
    # If it ain't, assume it is the label of a single seed variable.
    else:
        # First` get the blanket, including the seed variable.
        b = blanket(data, variables, alpha=alpha)
    # Avoid repetition.
    b = set(b)
    # Then subset the data.
    subdata = data[b]
    # Run causal discovery algorithm on blanket data only and return result.
    return algos[algo].predict(subdata)
