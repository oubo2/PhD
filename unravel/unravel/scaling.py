#!/bin/env python3
#
# scaling.py - to assess how various causal discovery algorithms scale.
#
# Author: Fjalar de Haan (fjalar.dehaan@unimelb.edu.au)
# Created: 2022-09-07
# Last modified: 2022-09-08
#

import time
import multiprocessing

import cdt
from cdt.data import AcyclicGraphGenerator as DAG

import numpy as np
import pandas as pd

# The algorithms to use.
algos = [ cdt.causality.graph.GES()
        , cdt.causality.graph.PC()
        , cdt.causality.graph.CAM()
        , cdt.causality.graph.LiNGAM() ]

# Synthetic data parameters (number of data points, scales and mechanism).
npoints = 10000
scales = [10, 50, 100, 500, 1000, 5000]
mechanism = 'polynomial'

# Columns for the dataframe.
columns = [ "algorithm"
          , "vertices [#]"
          , "data points [#]"
          , "runtime [s]" ]

def assess( algo=algos[0]
          , mechanism=mechanism
          , npoints=npoints
          , scale=scales[0] ):
    """Assess runtime of algorithm and return details as dataframe row."""

    # Let the terminal know what is going on.
    algoname = str(algo).split(sep='.')[3]
    print( "Running algorithm %s on %i vertices and %i data points."
            % (algoname, scale, npoints) )
    # Generate data.
    generator = DAG(mechanism, npoints=npoints, nodes=scale)
    data, graph = generator.generate()
    # Start the wall clock.
    start = time.time()
    # Do the thing.
    g = algo.predict(data)
    # Stop the wall clock and report to terminal.
    stop = time.time()
    runtime = stop-start
    print( "* %s on %i vertices and %i data points had runtime: %0.2f"
         % (algoname, scale, npoints, runtime) )
    # Create the dataframe holding the row of assessment data.
    row = [algoname, scale, npoints, runtime]
    rowframe = pd.DataFrame([row], columns=columns)
    # Return the row dataframe.
    return rowframe


def assess_sequential( algos=algos
                     , mechanism=mechanism
                     , npoints=npoints
                     , scales=scales ):
    """Return assessment of scaling for causal discovery algorithms."""

    # Prepare the dataframe to log results in.
    df = pd.DataFrame(columns=columns)

    # Fill the dataframe.
    for algo in algos:
        for scale in scales:
            # Get results for this algo and this scale.
            row = assess(algo, mechanism, npoints, scale)
            # Add the row to the log dataframe.
            df = df.append(row)
    # Return the dataframe
    return df


def assess_parallel( algos=algos
                   , mechanism=mechanism
                   , npoints=npoints
                   , scales=scales ):
    """Assess scaling of causal discovery algorithms parallelly."""

    # Prepare the dataframe to log results in.
    df = pd.DataFrame(columns=columns)

    # Create and fill pool and dataframe.
    rows = []
    pool = multiprocessing.Pool()
    for algo in algos:
        for scale in scales:
            # Get results for this algo and this scale.
            args = algo, mechanism, npoints, scale
            rows.append(pool.apply_async(assess, args))
    # Add the row to the log dataframe.
    for row in rows:
        df = df.append(row.get())
    # Return the dataframe.
    return df


def batchassess( algos=algos
               , mechanism=mechanism
               , npoints=npoints
               , scales=scales
               , parallel=True):
    """Return batch assessment of scaling for causal discovery algorithms."""
    if parallel:
        return assess_parallel(algos, mechanism, npoints, scales)
    else:
        return assess_sequential(algos, mechanism, npoints, scales)

if __name__ == '__main__':
    df = batchassess()
    df.to_csv("/home/fjalar/scaling.csv", index=False)
