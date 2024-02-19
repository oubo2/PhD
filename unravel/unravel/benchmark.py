#
# benchmark.py - tools for benchmarking causal discovery algorithms.
#
# Author: Fjalar de Haan (fjalar.dehaan@unimelb.edu.au)
# Created: 2023-06-02
# Last modified: 2023-06-02
#
from unravel.causal import *


import time
import multiprocessing

import cdt
from cdt.data import AcyclicGraphGenerator as DAG

import numpy as np
import pandas as pd


# Synthetic data parameters.
nvertices = [ 10, 50, 100, 500 ]
nrows = [ 100, 500, 1000 ]

def errors(bs): return [ not b for b in bs ]
def errorcount(bs): return sum(errors(bs))

def VHD(trial, truth, average=True):
    """Compute vertex-based hamming distances between `trial` and `truth`."""
    # Prepare a vertex->score dictionary.
    ds = {}
    # Check each vertex.
    for vertex in truth.nodes():
        score = 0
        # How many in-edges missed?
        score += errorcount([ trial.has_edge(*edge)
                              for edge in truth.in_edges(vertex) ])
        # How many in-edges made up?
        score += errorcount([ truth.has_edge(*edge)
                              for edge in trial.in_edges(vertex) ])
        # How many out-edges missed?
        score += errorcount([ trial.has_edge(*edge)
                              for edge in truth.out_edges(vertex) ])
        # How many out-edges made up?
        score += errorcount([ truth.has_edge(*edge)
                              for edge in trial.out_edges(vertex) ])
        # Provide the score as well as the true number of edges.
        ds[vertex] = score, truth.degree(vertex)
    # Deliver.
    if average:
        values = [ t[0] for t in ds.values() ]
        return sum(values) / len(ds)
    else:
        return ds

def precision(trial, truth, average=True):
    """Return fraction of true edges and all edges in `trial`, by vertex."""
    # Prepare a vertex->score dictionary.
    ds = {}
    # Check each vertex.
    for vertex in truth.nodes():
        score = 0
        # How many true in-edges?
        score += sum([ trial.has_edge(*edge)
                       for edge in truth.in_edges(vertex) ])
        # How many true out-edges?
        score += sum([ trial.has_edge(*edge)
                       for edge in truth.out_edges(vertex) ])
        # Compute the precision.
        # TODO: Check the below --- trial.degree?!
        # if score == 0 and trial.degree(vertex) == 0:
            # ds[vertex] = 1.0
        if trial.degree(vertex) > 0:
            ds[vertex] = score / trial.degree(vertex)
    # Deliver.
    if average:
        return sum(ds.values()) / len(ds)
    else:
        return ds

def recall(trial, truth, average=True):
    """Return fraction of all true edges found in `trial`, by vertex."""
    # Prepare a vertex->score dictionary.
    ds = {}
    # Check each vertex.
    for vertex in truth.nodes():
        score = 0
        # How many true in-edges?
        score += sum([ trial.has_edge(*edge)
                       for edge in truth.in_edges(vertex) ])
        # How many true out-edges?
        score += sum([ trial.has_edge(*edge)
                       for edge in truth.out_edges(vertex) ])
        # Compute the recall.
        if score == 0 and truth.degree(vertex) == 0:
            ds[vertex] = 1.0
        else:
            ds[vertex] = score / truth.degree(vertex)
    # Deliver.
    if average:
        return sum(ds.values()) / len(ds)
    else:
        return ds

def rel_edge_error(trial, truth):
    """Return average and std dev of edge error per true edge."""
    average = np.mean([ score[0]/score[1]
                        for (vertex, score) in VHD(g, graph).items()
                        if score[1] != 0 ] # Don't count isolated vertices.
                     )
    stddev = np.std([ score[0]/score[1]
                      for (vertex, score) in VHD(g, graph).items()
                      if score[1] != 0 ] # Don't count isolated vertices.
                   )
    return average, stddev

def benchmark( algolist            # The list (strings) of algorithms to use.
             , nvertices           # How many vertices in the generated graphs.
             , nrows               # How many rows of generated data.
             , iterations=11       # Number of iterations to average over.
             , mechanism='linear'  # Causal 'mechanism' to use.
             , noise='gaussian'    # Distribution of the noise to use.
             , chunksize=None      # Partition column set in chunks this size.
             , target=None         # Focus on this variable.
             , returndata=False ): # Also return truth/trial graphs and data.
    """Compute a suite of benchmarks."""
    # Prepare a metric->score dictionary.
    benchmarks = {}
    # Initialise variables.
    prc =  rec = vhd = shd = sid = 0.0
    # Prepare empty lists for graphs and datasets.
    truths = []
    datas = []
    trials = []
    # Iterate to get credible statistics.
    for i in range(iterations):
        # Generate the random causal network.
        data, truth = generate( mechanism
                              , noise=noise
                              , nvertices=nvertices
                              , nrows=nrows)
        j = i+1
        print("Discovering causal graphs for iteration %i." % j)
        trial = discover(algolist, data, chunksize, target)
        # Archive the graph and data if asked.
        if returndata:
            truths.append(truth)
            trials.append(trial)
            datas.append(data)
        # Calculate the statistics.
        if target is None:
            prc += precision(trial, truth)
            rec += recall(trial, truth)
        else:
            prc_dict = precision(trial, truth, average=False)
            if target in prc_dict:
                prc += prc_dict[target]
            else:
                prc -= 1000*iterations # TODO: Deal more elegantly with this.
            rec_dict = recall(trial, truth, average=False)
            if target in rec_dict:
                rec += rec_dict[target]
            else:
                rec -= 1000*iterations # TODO: Deal more elegantly with this.
        vhd += VHD(trial, truth)
        shd += SHD(trial, truth)
        sid += SID(trial, truth)
    # Do the averaging.
    benchmarks["precision"] = prc / iterations
    benchmarks["recall"] = rec / iterations
    benchmarks["VHD"] = vhd / iterations
    benchmarks["SHD"] = shd / iterations
    benchmarks["SID"] = sid / iterations
    if returndata:
        benchmarks["truths"] = truths
        benchmarks["trials"] = trials
        benchmarks["datas"] = datas
    # Deliver.
    return benchmarks

def discover(algolist, data, chunksize=None, target=None, intersected=True):
    # In case just one algo is passed, put it in a list anyway.
    if type(algolist) == str: algolist = [algolist]
    # Prepare an empty list to hold the discovered causal graphs' edges.
    edgesets = []
    # Prepare an empty dictionary to keep the algo => graph pairs.
    graphs = {}
    # Only do it chunkedly if asked and necessary.
    if chunksize is None or chunksize >= data.shape[0]:
        # Add causal graphs returned by each algorithm.
        for algo in algolist:
            print("Running %s algorithm." % algo)
            graph = algos[algo].predict(data)
            # Add the edgeset of the discovered graph to the list.
            edgesets.append(graph.edges)
            # Add the graph to the dictionary.
            graphs[algo] = graph
    else:
        # Add causal graphs returned by each algorithm.
        for algo in algolist:
            print("Running %s algorithm in chunks of %i." % ( algo
                                                            , chunksize))
            chunks = partrand(data.columns, chunksize, var=target)
            chunkgraphs = [ algos[algo].predict(data[chunk])
                            for chunk in chunks ]
            graph = nx.compose_all(chunkgraphs)
            # Add the edgeset of the discovered graph to the list.
            edgesets.append(graph.edges)
            # Add the graph to the dictionary.
            graphs[algo] = graph
    # Collect the intersection of the edge-sets.
    sharededges = set.intersection(*map(set, edgesets))
    # Construct the graph from the intersection.
    graph = nx.DiGraph()
    graph.add_nodes_from(data.columns)
    graph.add_edges_from(sharededges)
    # Deliver either the intersection or the dictionary of all graphs.
    if intersected:
        return graph
    else:
        return graphs

def generate( mechanism='linear'
            , noise='gaussian'
            , nvertices=nvertices
            , nrows=nrows ):
    """Generate synthetic data and corresponding 'ground-truth' causal graph."""
    generator = DAG(mechanism, noise=noise, nodes=nvertices, npoints=nrows)
    data, truth = generator.generate()
    return data, truth


def tonetworkx(clgraph):
    """Convert a Causal-Learn graph object to a NetworkX graph object."""
    # Populate the internal NetworkX DiGraph object.
    clgraph.to_nx_graph()
    # Get it out of the Causal-Learn graph object.
    g = clgraph.nx_graph
    # Create a old label to new label dictionary.
    d = { vertex: 'V' + str(vertex) for vertex in gprime.nodes() }
    # Relabel and overwrite graph.
    g = nx.relabel_nodes(g, d)
    # Deliver.
    return g










