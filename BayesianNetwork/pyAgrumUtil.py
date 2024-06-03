import pyAgrum as gum
from util import *
import pyAgrum.lib.notebook as gnb 

def getPyAgrumBayesNet(df_alarm, nodes, edges):
    #could cauclate nodes edges here but not ideal since there are different methods
    bn=gum.BayesNet('TEP')
    for node in nodes:
        bn.add(gum.LabelizedVariable(node, node, df_alarm[node].nunique()))
    for edge in edges:
        bn.addArc(edge[0], edge[1])
    for node in nodes:
        node_id = bn.cpt(node).names[0]
        parents = list(bn.cpt(node).names[1:])
        outcomes = sorted(list(df_alarm[node].value_counts().index))
        if len(parents) == 0:
            probability = [calculate_probability(df_alarm, node_id, outcome) for outcome in outcomes]
            bn.cpt(node_id)[:] = probability
        else:
            parents_outcomes = [sorted(list(df_alarm[p].value_counts().index)) for p in parents]
            parents_outcomes = generate_combinations(parents_outcomes)
            for parents_outcome in parents_outcomes:
                dic = {}
                for i in range(len(parents_outcome)):
                    dic[parents[i]] = parents_outcome[i]
                cpt = [calculate_conditional_probability(df_alarm, node_id, outcome, parents, parents_outcome) for outcome in outcomes]
                if sum(cpt) == 0:
                    cpt = [1/len(cpt)] * len(cpt)
                bn.cpt(node_id)[dic] = cpt
    return bn