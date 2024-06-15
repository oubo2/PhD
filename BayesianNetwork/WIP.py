"""
from pathlib import Path
#from natsort import natsorted
data_path = Path('.', 'data/3Wdata/data/data')
events_names = {0: 'Normal',
                1: 'Abrupt Increase of BSW',
                2: 'Spurious Closure of DHSV',
                3: 'Severe Slugging',
                4: 'Flow Instability',
                5: 'Rapid Productivity Loss',
                6: 'Quick Restriction in PCK',
                7: 'Scaling in PCK',
                8: 'Hydrate in Production Line'
               }
vars = ['P-PDG',
        'P-TPT',
        'T-TPT',
        'P-MON-CKP',
        'T-JUS-CKP',
        'P-JUS-CKGL',
        'T-JUS-CKGL',
        'QGL']
columns = ['timestamp'] + vars + ['class']
rare_threshold = 0.01
normal_class_code = 0
undesirable_event_code = 1      # Undesirable event of interest
downsample_rate = 60            # Adjusts frequency of sampling to the dynamics 
                                # of the undesirable event of interest
sample_size_default = 60        # In observations (after downsample)
sample_size_normal_period = 5   # In observations (after downsample)
max_samples_per_period = 15     # Limitation for safety

def class_and_file_generator(data_path, real=False, simulated=False, drawn=False):
    for class_path in data_path.iterdir():
        if class_path.is_dir():
            class_code = int(class_path.stem)
            for instance_path in class_path.iterdir():
                if (instance_path.suffix == '.csv'):
                    if (simulated and instance_path.stem.startswith('SIMULATED')) or \
                       (drawn and instance_path.stem.startswith('DRAWN')) or \
                       (real and (not instance_path.stem.startswith('SIMULATED')) and \
                       (not instance_path.stem.startswith('DRAWN'))):
                        yield class_code, instance_path

def load_instance(instance_path):
    try:
        well, instance_id = instance_path.stem.split('_')
        df = pd.read_csv(instance_path, sep=',', header=0)
        assert (df.columns == columns).all(), 'invalid columns in the file {}: {}'\
            .format(str(instance_path), str(df.columns.tolist()))
        return df
    except Exception as e:
        raise Exception('error reading file {}: {}'.format(instance_path, e))
        
def load_and_downsample_instances(instances, downsample_rate, source, instance_id):
    df_instances = pd.DataFrame()
    for _, row in instances.iterrows():
        _, instance_path = row
        df = load_instance(instance_path).iloc[::downsample_rate, :]
        df['instance_id'] = instance_id
        instance_id += 1
        df_instances = pd.concat([df_instances, df])
    df_instances['source'] = source
    return df_instances.reset_index(drop=True), instance_id

def get_instances_with_undesirable_event(data_path, undesirable_event_code,
                                         real, simulated, drawn):
    instances = pd.DataFrame(class_and_file_generator(data_path,
                                                      real=real,
                                                      simulated=simulated, 
                                                      drawn=drawn),
                             columns=['class_code', 'instance_path'])
    idx = instances['class_code'] == undesirable_event_code
    return instances.loc[idx].reset_index(drop=True)

normal_instances = get_instances_with_undesirable_event(data_path, 
                                                      normal_class_code,
                                                      real=True,
                                                      simulated=False, 
                                                      drawn=False)

faulty_instances = get_instances_with_undesirable_event(data_path, 
                                                      undesirable_event_code,
                                                      real=True,
                                                      simulated=False, 
                                                      drawn=False)

instance_id = 0
df_normal_instances, instance_id  = load_and_downsample_instances(normal_instances,
                                                                downsample_rate,
                                                                'real', 
                                                                instance_id)
df_faulty_instances, instance_id  = load_and_downsample_instances(faulty_instances,
                                                                downsample_rate,
                                                                'real', 
                                                                instance_id)

df_test_instances = df_normal_instances.drop(['timestamp', 'instance_id', 'source', 'class', 'T-JUS-CKGL'], axis=1)
df_test_instances = df_test_instances.fillna(method='pad')
#df_test_instances.isna().sum()
df_test1_instances = df_faulty_instances.drop(['timestamp', 'instance_id', 'source', 'class', 'T-JUS-CKGL'], axis=1)
df_test1_instances = df_test1_instances.fillna(method='pad')

std_scaler = StandardScaler()
df_scaled = std_scaler.fit_transform(df_test_instances.to_numpy())
df_scaled = pd.DataFrame(df_scaled, columns=list(df_test_instances.columns))
df_scaled1 = std_scaler.transform(df_test1_instances.to_numpy())
df_scaled1 = pd.DataFrame(df_scaled1, columns=list(df_test_instances.columns))


df_alarm = alarm_df(df_scaled, df_scaled1)
score_Table = scoreTable(df_alarm, transferEntropyScore, 10)
print(score_Table)
boolTable = probabilityToBool(score_Table, 0.0, True)
edges = getTableEdges(boolTable, df_alarm)
nodes = getNodesFromEdges(edges)
print("no of nodes " + str(len(nodes)) + ", no of edges " + str(len(edges)))
removedEdges = removeCycles(edges, score_Table, df_alarm)
print(removedEdges)
for edge in removedEdges:
    edges.remove(edge)
drawBN(edges)
"""

"""
Maximum Likelihood estimation
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
data = pd.DataFrame(df_alarm)
model = BayesianNetwork(edges)
cpd_A = MaximumLikelihoodEstimator(model, data).estimate_cpd('xmeas_16')
print(cpd_A)
"""