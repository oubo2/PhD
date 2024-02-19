#!/bin/env python3
#
# unravel.py - causal modelling tools loader script.
#
# Author: Fjalar de Haan (fjalar.dehaan@unimelb.edu.au)
# Created: 2023-03-8
# Last modified: 2023-04-25
#
from unravel import *
# from unravel.hilda import *

import pyreadr

train_normal_path = '/TEP2017R/TEP_FaultFree_Training.RData'
train_faulty_path = '/TEP2017R/TEP_Faulty_Training.RData'
 
test_normal_path = '/TEP2017R/TEP_FaultFree_Testing.RData'
test_faulty_path = '/TEP2017R/TEP_Faulty_Testing.RData'
 
train_normal_complete = pyreadr.read_r(train_normal_path)['fault_free_training']
#train_faulty_complete = pyreadr.read_r(train_fault_path)['faulty_training']
 
#test_normal_complete = pyreadr.read_r(test_normal_path)['fault_free_testing']
test_faulty_complete = pyreadr.read_r(test_faulty_path)['faulty_testing']
df_train = train_normal_complete[train_normal_complete.simulationRun==1].iloc[:,3:]
 
fig, ax = plt.subplots(13,4,figsize=(30,50))

g = algos['GES'].predict(data)
#g = causal_blanket(data, variables, algo=ALGORITHM)
#gprint()

'''def cli_args():
    """Parse `argv` interpreter-agnostically. Return non-trivial arguments."""
    argv = os.sys.argv
    fname = os.path.basename(__file__)
    print(fname)
    for arg in argv:
        if arg.endswith(fname):
            index = argv.index(arg) + 1
    if len(argv) > index:
        return argv[index:]
    else:
        return []

def run_algo_on_hilda(algo):
    """Run algorithm `algo` on HILDA and write causal graph to pickle."""
    print("Running %s algorithm." % algo)
    g = algos[algo].predict(hilda)
    pname = "graph-" + algo + ".pickle"
    with open(pname, "wb") as f:
        print("Writing output of %s to %s." % (algo, pname))
        pickle.dump(g, f)

def run_algo_on_hilda1k(algo):
    """Run algorithm `algo` on HILDA and write causal graph to pickle."""
    print("Running %s algorithm." % algo)
    g = algos[algo].predict(hilda1k)
    pname = "graph-" + algo + "-hilda1k" + ".pickle"
    with open(pname, "wb") as f:
        print("Writing output of %s to %s." % (algo, pname))
        pickle.dump(g, f)

def run_algo_on_hilda100(algo):
    """Run algorithm `algo` on HILDA and write causal graph to pickle."""
    print("Running %s algorithm." % algo)
    g = algos[algo].predict(hilda1k)
    pname = "graph-" + algo + "-hilda100" + ".pickle"
    with open(pname, "wb") as f:
        print("Writing output of %s to %s." % (algo, pname))
        pickle.dump(g, f)

def run_algo_on_candidates(algo):
    with open("../analysis-hilda/candidates-20230406.pickle", "rb") as f1:
        cs = pickle.load(f1)
    candidates = cs + bcols
    print("Running %s algorithm." % algo)
    g = algos[algo].predict(hilda1k[candidates])
    pname = "graph-" + algo + "-hilda100" + ".pickle"
    with open(pname, "wb") as f2:
        print("Writing output of %s to %s." % (algo, pname))
        pickle.dump(g, f2)

def load_analysis():
    global g, h, dg, dh, dcauses, deffects, catcauses, cateffects
    with open( "/home/fjalar/pCloudDrive/archive/academia/projects/"
               "future-of-work/analysis-hilda/"
               "graph-GIES-hilda100-bcols.pickle", "rb") as f:
        g = pickle.load(f)
    with open( "/home/fjalar/pCloudDrive/archive/academia/projects/"
               "future-of-work/analysis-hilda/"
               "graph-GIES-hilda100-tjbmsall.pickle", "rb") as f:
        h = pickle.load(f)
    dg = { col: meta.column_names_to_labels[col] for col in g.nodes() }
    dh = { col: meta.column_names_to_labels[col] for col in h.nodes() }
    gplint(nx.relabel_nodes(causes(g, 'tjbmsall'), dg))
    gplint(nx.relabel_nodes(causes(h, 'tjbmsall'), dh))
    dcauses= nx.shortest_path_length(g, target='tjbmsall')
    deffects = nx.shortest_path_length(g, source='tjbmsall')
    cateffects = { cat: np.mean( [ deffects[v] for v in contractions[cat]
                                                if v in deffects ] )
                  for cat in contractions }
    catcauses = { cat: np.mean( [ dcauses[v] for v in contractions[cat]
                                              if v in dcauses ] )
                  for cat in contractions }

def run_isco(algo='GIES', isco=None):
    if isco is None:
        print("Error: ISCO code not provided.")
    else:
        print("Subsetting HILDA to ISCO code %i." % isco)
        h = hilda_by_isco(isco)
        print("Getting candidate causes and effects.")
        cs = candidates(bcols, h)
        print("Discovering causal graph.")
        g = algos[algo].predict(h[cs + bcols])
        # Write the causal graph to a pickle.
        pname = "graph-isco" + str(isco) + ".pickle"
        with open(pname, "wb") as f:
            print("Writing causal graph to %s." % pname)
            pickle.dump(g, f)
        return g

def run_stratified(algo='GIES', iscos=iscover100):
    # Prepare a dictionary for the causal graphs.
    d = {}
    # It is okay to pass a single code not wrapped in a list.
    if type(iscos) != list: iscos = [iscos]
    # Put causal graph in dictionary keyed by ISCO code.
    for isco in iscos:
        d[isco] = run_isco(algo=algo, isco=isco)
    # Deliver.
    with open("graphs-by-isco-dict.pickle", "wb") as f: pickle.dump(d, f)
    return d

def run_stratified_parallel(algo='GIES', iscos=iscover100):
    # Set up the multiprocessing facilities.
    pool = multiprocessing.Pool()
    # Collect the multiprocessing pre-results.
    d = { isco: pool.apply_async(run_isco, (algo, isco)) for isco in iscos }
    # Then extract the actual causal graphs and return them.
    return { isco: d[isco].get() for isco in iscos }

def run_isco_colsampled(algo='GIES', isco=None, ncols=10, niters=10):
    if isco is None:
        print("Error: ISCO code not provided.")
    else:
        print("Subsetting HILDA to ISCO code %i." % isco)
        h = hilda_by_isco(isco)
        gs = []
        i = 0
        while len(gs) < niters:
            hsample = h.sample(n=ncols, axis=1, random_state=i)
            if 'ujbmsall' in hsample:
                print("Discovering causal graph number %i." % len(gs))
                g = algos[algo].predict(hsample)
                gs.append(g)
                print("Appended graph.")
            i += 1
        composition = gs[0]
        for g in gs[1:]:
            composition = nx.compose(composition, g)

        return gs, composition


if __name__ == '__main__':
    args = cli_args()
    if len(args) > 0:
        print(args)
        if args[0] == 'blankets':
            # memory_limit()
            bs = blankets(hilda, list(cols.keys()), parallel=False)
            # bs = blankets(hilda, list(cols.keys()), parallel=True)
            with open("blankets.pickle", "wb") as f: pickle.dump(bs, f)
        else:
            run_stratified_parallel(args[0])
    else:
        pass'''
