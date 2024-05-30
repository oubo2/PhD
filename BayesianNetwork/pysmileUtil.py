from util import *
import pysmile
import pysmile_license

def create_cpt_node(net, id, name, outcomes):
    handle = net.add_node(pysmile.NodeType.CPT, id)
    net.set_node_name(handle, name)
    initial_outcome_count = net.get_outcome_count(handle)
    for i in range(0, initial_outcome_count):
        net.set_outcome_id(handle, i, outcomes[i])
    for i in range(initial_outcome_count, len(outcomes)):
        net.add_outcome(handle, outcomes[i])    
    return handle

def print_posteriors(net, node_handle):
    node_id = net.get_node_id(node_handle)
    if net.is_evidence(node_handle):      
        print(f"{node_id} has evidence set ({net.get_evidence_id(node_handle)})") 
    else :           
        posteriors = net.get_node_value(node_handle)           
        for i in range(0, len(posteriors)):               
            print(f"P({node_id}={net.get_outcome_id(node_handle, i)})={posteriors[i]}")    

def print_all_posteriors(net):        
    for handle in net.get_all_nodes():            
        print_posteriors(net, handle)        
        
def change_evidence_and_update(net, node_id, outcome_id):        
    if outcome_id is not None:           
        net.set_evidence(node_id, outcome_id)       
    else:            
        net.clear_evidence(node_id)        
    net.update_beliefs()        
    print_all_posteriors(net)        
    print()
    
def print_node_info(net, node_handle):     
    print(f"Node id/name: {net.get_node_id(node_handle)}/{net.get_node_name(node_handle)}")      
    print(f"  Outcomes: {' '.join(net.get_outcome_ids(node_handle))}")          
    parent_ids = net.get_parent_ids(node_handle)       
    if len(parent_ids) > 0:            
        print(f"  Parents: {' '.join(parent_ids)}")        
    child_ids = net.get_child_ids(node_handle)       
    if len(child_ids) > 0:           
        print(f"  Children: {' '.join(child_ids)}")          
    print_cpt_matrix(net, node_handle)  

def print_cpt_matrix(net, node_handle):       
    cpt = net.get_node_definition(node_handle)    
    parents = net.get_parents(node_handle)     
    dim_count = 1 + len(parents)             
    dim_sizes = [0] * dim_count     
    for i in range(0, dim_count - 1):  
        dim_sizes[i] = net.get_outcome_count(parents[i])    
    dim_sizes[len(dim_sizes) - 1] = net.get_outcome_count(node_handle)  
    coords = [0] * dim_count        
    for elem_idx in range(0, len(cpt)):         
        index_to_coords(elem_idx, dim_sizes, coords)          
        outcome = net.get_outcome_id(node_handle, coords[dim_count - 1])      
        print(f"    P({outcome}", end="")                
        if dim_count > 1:          
            print(" | ", end="")      
            for parent_idx in range(0, len(parents)):         
                if parent_idx > 0:                      
                    print(",", end="")                 
                parent_handle = parents[parent_idx]
                print(f"{net.get_node_id(parent_handle)}="         
                      + f"{net.get_outcome_id(parent_handle, coords[parent_idx])}", end="")     
        prob = cpt[elem_idx]           
        print(f")={prob}")  
        
def index_to_coords(index, dim_sizes, coords):       
    prod = 1      
    for i in range(len(dim_sizes) - 1, -1, -1):     
        coords[i] = int(index / prod) % dim_sizes[i]        
        prod *= dim_sizes[i]

# Function to calculate the conditional probability P(X|Y)
def calculate_conditional_probability(df, var_X, value_X, parents, parents_values):
    conditions = [df[parents[i]] == parents_values[i] for i in range(len(parents))]
    #print(var_X, value_X, parents, parents_values)
    condition = df[parents[0]] == parents_values[0]
    #print(condition.value_counts())
    for i in range(len(parents)):
        df = df[df[parents[i]] == parents_values[i]]
    total_count = len(df)
    if total_count == 0:
        return 0
    joint_count = len(df[df[var_X] == value_X])
    return joint_count / total_count

def convert_outcome_to_value(outcome):
    if outcome[:5] == "State":
        outcome = int(outcome[5:])
    return outcome

def get_outcomes(net, node):
    outcomes = net.get_outcome_ids(node)
    return [convert_outcome_to_value(outcome) for outcome in outcomes]

def getBayesianNet(df_alarm):
    net = pysmile.Network()
    TETable = transferEntropyTable(df_alarm, 4, 0.08)
    feature_list = list(df_alarm.columns)
    edges = getTableEdges(TETable, feature_list)
    nodes = getNodesFromEdges(edges)
    
    net_nodes = []
    for node in nodes:
        # Bayesian net can not handle list of int as outcomes, only list of strings
        o = [str(i) for i in sorted(list(df_alarm[node].value_counts().index))]
        net_nodes.append(create_cpt_node(net, node, node, o))

    for edge in edges:
        net.add_arc(edge[0], edge[1])
    

    for node in net_nodes:
        nodeDef = []
        node_id = net.get_node_id(node)
        parents = net.get_parent_ids(node)
        outcomes = get_outcomes(net, node)
        
        if len(parents) == 0:
            nodeDef = [calculate_probability(df_alarm, node_id, outcome) for outcome in outcomes]
        else:
            parents_outcomes = [get_outcomes(net, node) for p in parents]
            parents_outcomes = generate_combinations(parents_outcomes)
            
            nodeDef = [calculate_conditional_probability(df_alarm, node_id, outcome, parents, i) 
                   for i in parents_outcomes for outcome in outcomes]
            print(nodeDef)
        net.set_node_definition(node, nodeDef)
    

    return net