#!/bin/env python3
#
# hilda.py - for (pre)processing of HILDA data.
#
# Author: Fjalar de Haan (fjalar.dehaan@unimelb.edu.au)
# Created: 2022-09-12
# Last modified: 2023-05-05
#

import pickle, os

import numpy as np
import pandas as pd
import networkx as nx
import pyreadstat

# Path strings to 20th ('t') wave of HILDA data set.
project_path = ( "/home/fjalar/pCloudDrive/"
                 "archive/academia/projects/future-of-work/" )
# hilda_spss_path = "/server/data/hilda2020/spss-200c/Combined t200c.sav"
# hilda_spss_path = "/server/data/hilda2021/spss-210c/Combined u210c.sav"
hilda_spss_path = project_path+"data/hilda2021/spss-210c/Combined u210c.sav"
hilda_pickle_path = project_path+"data/hilda2020/hilda-combined-t200c.pickle"
raw_pickle_path = project_path+"data/hilda2020/hilda-combined-t200c-raw.pickle"

def clean(raw, fill='mode'):
    """Clean HILDA data."""
    # Exclude `object` cols containing wave ids, dates and other irrelevantia.
    cleaned = raw.select_dtypes(include='float64').copy()
    # Drop columns with only NaNs.
    cleaned.dropna(axis='columns', how='all', inplace=True)
    # Replace ramaining NaNs.
    if fill == 'mean':
        # Replace NaNs with mean values --- this messes up variables like `sex`.
        cleaned.fillna(cleaned.mean().to_dict(), inplace=True)
    else:
        # Replace NaNs with most common values.
        d = cleaned.mode().to_dict()
        replacements = {key: val[0] for (key, val) in d.items()}
        cleaned.fillna(replacements, inplace=True)
    # Drop columns with only one value.
    cols = [ col for col in cleaned.columns
                 if pd.unique(cleaned[col]).shape[0]==1 ]
    cleaned.drop(columns=cols, inplace=True)
    return cleaned

def stats(data):
    """Some statistics of the data."""
    rowlabels = [ "nunique"
                , "min"
                , "mean"
                , "max"
                , "std" ]
    df = pd.DataFrame(columns=data.columns, index=rowlabels)
    for col in data.columns:
        df.loc["nunique", col] = pd.unique(data[col]).shape[0]
    df.loc["min"] = data.min()
    df.loc["mean"] = data.mean()
    df.loc["max"] = data.max()
    df.loc["std"] = data.std()
    return df

jcols = { 'ulosatsf': 'Life satisfaction level'
        , 'ujbhruc': 'Combined per week usually worked in all jobs'
        , 'uhiwsfei': 'Imputed financial year gross wages and salary'
        , 'uedhists': 'highest level of education'
        , 'uhhda10': 'SEIFA decile of socio-economic disadvantage'
        }

fcols = { # Basic demographics.
          'uhgage': 'DV Age last birthday at June 30'
        , 'uhgsex': 'Sex'
        , 'umrcurr': 'Marital status'
        , 'uedhigh1': 'Highest education level achieved'
        # , 'uedlhqn': 'Highest education level' # Not in Wave 21.
        , 'ues': 'Employment  status'
        , 'uhhda10': 'SEIFA decile of socio-economic disadvantage'
          # Work-related factors.
        , 'ujbmsall': 'Overall job satisfaction'
        , 'ujbmsflx': 'Flexibility to balance work/life satisfaction'
        , 'uesdtl': 'Labour force status detailed'
        , 'ujbmshrs': 'Hours working satisfaction'
        , 'uwscei': 'Current gross weekly income'
        , 'uwsfei': 'Financial year gross income'
        , 'ujbmspay': 'Total pay satisfaction'
        , 'ujbmhruc': 'Hours per week in main job'
        , 'ujbhruc': 'Hours per week in all jobs'
        , 'ujbhrcpr': 'Preference to work fewer/same/more hours'
        , 'ujbmssec': 'Job security satisfaction'
        , 'ujbmswrk': 'Work itself satisfaction'
        , 'ulosateo': 'Employment opportunities satisfaction'
        , 'ujbmo62': 'Occupation code 2-digit ANZSCO'
        , 'ujbmcnt': 'Employment contract (current job)'
        , 'ujbmh': 'Any usual working hours worked from home'
        , 'ujbmlha': 'Main job location varies'
        # (Not in index) , 'ujbmlkm': 'Main job location distance from home'
          # Health-related factors.
        , 'ulosatyh': 'Health satisfaction'
        , 'ugh1': 'Self-assessed health'
        , 'ughgh': 'SF-36 general health'
        , 'ughmh': 'SF-36 mental health'
        , 'ujomms': 'Job is more stressful than I had ever imagined'
        , 'ulosat': 'Life satisfaction'
        }

cols = {
       # 'uhgage': 'Age (approx.)'
       # , 'uhgsex': 'Sex'
       # , 'umrcurr': 'Marital status'
       # , 'uedhigh1': 'Highest education level achieved'
       # , 'ues': 'Employment  status'
        'uhhda10': 'SEIFA decile of socio-economic disadvantage'
       , 'ujbmsall': 'Overall job satisfaction'
       , 'ughmh': 'SF-36 mental health'
       }

bcols = [ 'ujomus'
        # , 'uskcjed' # Not in Wave 21.
        , 'ujomcd'
        , 'ujomfast'
        , 'ujomms'
        , 'ujompi'
        , 'ujomtime'
        , 'ujomwi'
        , 'ujbempt'
        # , 'ujbmcntr' # Only in Wave 1.
        , 'ujbmploj'
        , 'ujbmssec'
        , 'ujbocct'
        , 'ujomsf'
        , 'ujomwf'
        , 'ulosateo'
        # , 'ujoskill' # Only in Wave 5.
        , 'ujomns'
        , 'ujomls'
        , 'ujomini'
        , 'ujowpcc'
        # , 'ujowpcr' # Only in  1 < Wave < 20.
        , 'ujowpptw'
        # , 'ujowpuml' # Only in 1 < Wave < 11.
        , 'ujowppml'
        # , 'ujowppnl' # Only in 1 < Wave < 11.
        , 'ujompf'
        , 'uwscei'
        , 'uwscg'
        , 'uwsfei'
        , 'uwsfes'
        , 'ujbmswrk'
        # , 'ujonomfl' # Only in Wave 5.
        # , 'ujoserve' # Only in Wave 5.
        # , 'ujosoc' # Only in Wave 5.
        # , 'ujosat' # Only in Wave 5.
        , 'ujbmsall'
        # , 'ujostat' # Only in Wave 5.
        # , 'ujonovil' # Only in Wave 5.
        , 'ujomdw'
        , 'ujomrpt'
        , 'ujomvar'
        , 'ujbmagh'
        , 'ujbmh'
        , 'ujbmhl'
        # , 'ujbmhrh' # Only in Wave 1.
        # , 'ujbmhrha' # Not enough rows, drops out after sampling.
        , 'ujbmhruc'
        , 'ujbmsl'
        , 'ujowpfx'
        , 'ujowphbw'
        , 'ujombrk'
        , 'ujomdw'
        , 'ujomfd'
        , 'ujomflex'
        , 'ujomfw'
        # , 'ujbtremp' # Only in 2 <Wave < 7
        , 'ujbmsflx'
        , 'ujbhruc'
        # , 'ujbhru' # Only in Wave 1.
        # , 'ujbmhruw' # Not enough rows, drops out after sampling.
        # , 'uatwkhpj' # Only in Wave 1.
        , 'ulosat'
        , 'ujbmshrs'
        , 'ujbmspay'
        , 'ulosatfs'
        , 'ulosatft'
        , 'ujbnewjs'
        , 'ujompi'
        , 'ulosatyh' ]

# Concepts and their HILDA variable representatives.
concepts = { 'age': ['uhgage']
           , 'sex': ['uhgsex']
           , 'education': ['uedhigh1']
           , 'seifa': ['uhhsed10']
           # /\ General  demographic concepts -> variables.
           #
           # \/ FoW concepts -> variables. 
           , 'authority': ['ujomls']
           , 'autonomy': ['ujomini', 'ujomdw']
           , 'career and skill development (growth)': ['ujomns']
           , 'career opportunities': ['ulosateo']
           , 'flexible work practices': [ 'ujowphbw'
                                        , 'ujbmhruc'
                                        , 'ujbmh'
                                        , 'ujbmhl'
                                        , 'ujomfd'
                                        , 'ujbmsl'
                                        , 'ujombrk'
                                        , 'ujowpfx'
                                        , 'ujbmagh'
                                        , 'ujbmhrha'
                                        , 'ujomfw'
                                        , 'ujomflex' ]
           , 'income': [ # 'ujowppml' # Paid maternity leave.
                         'uwsfei'
                       , 'ujompf'
                       , 'uwsfes'
                       , 'uwscei'
                       , 'uwscg' ]
           , 'job attitudes': ['ulosatfs', 'ulosatft', 'ujbmspay', 'ujbmshrs']
           , 'job demand (stress)': [ 'ujomms'
                                    , 'ujompi'
                                    , 'ujomwi'
                                    , 'ujomfast'
                                    , 'ujomcd'
                                    , 'ujomtime' ]
           , 'job resources': ['ujowpptw', 'ujowpcc']
           , 'job satisfaction': ['ujbmsall', 'ujbmswrk']
           , 'life satisfaction': ['ulosat']
           , 'long term employment and job security': [ 'ujbmploj'
                                                      , 'ujbempt'
                                                      , 'ujbocct'
                                                      , 'ujomwf'
                                                      , 'ujomsf'
                                                      , 'ujbmssec' ]
           , 'personality': [ 'upnextrv'
                            , 'upnagree'
                            , 'upnconsc'
                            , 'upnemote'
                            , 'upnopene' ]
           , 'skill-job fit': ['ujomus', 'ujsrealt']
           , 'turnover intentions': ['ujbnewjs']
           , 'variety': ['ujomrpt', 'ujomvar']
           , 'well being (mental and physical)': ['ulosatyh']
           , 'mental health': [ 'ujbumnt'
                              , 'uhemirh'
                              , 'uheomi'
                              , 'uhepmomi'
                              , 'uhecpmhp'
                              , 'uhemirhn'
                              , 'ughmh' ]
           # , 'work engagement': ['ujbmswrk']
           , 'work-life balance': ['ujbmsflx']
           , 'working hours': ['ujbmhruw', 'ujbhruc'] }

contractions = { 'authority': ['ujomls']
               , 'autonomy': ['ujomini']
               , 'career and skill development (growth)': [ 'ujoskill'
                                                          , 'ujomns']
               , 'career opportunities': ['ulosateo']
               , 'flexible work practices': [ 'ujbmagh'
                                            , 'ujbmh'
                                            , 'ujbmhl'
                                            , 'ujbmhrh'
                                            , 'ujbmhrha'
                                            , 'ujbmhruc'
                                            , 'ujbmsl'
                                            , 'ujowpfx'
                                            , 'ujowphbw'
                                            , 'ujombrk'
                                            , 'ujomdw'
                                            , 'ujomfd'
                                            , 'ujomflex'
                                            , 'ujomfw' ]
               , 'income': [ 'ujowppml'
                           , 'ujowppnl'
                           , 'ujompf'
                           , 'uwscei'
                           , 'uwscg'
                           , 'uwsfei'
                           , 'uwsfes' ]
               , 'job attitudes': [ 'ujbmshrs'
                                  , 'ujbmspay'
                                  , 'ulosatfs'
                                  , 'ulosatft' ]
               , 'job demand (stress)': [ 'ujomcd'
                                        , 'ujomfast'
                                        , 'ujomms'
                                        , 'ujompi'
                                        , 'ujomtime'
                                        , 'ujomwi' ]
               , 'job resources': [ 'ujowpcc'
                                  , 'ujowpcr'
                                  , 'ujowpptw'
                                  , 'ujowpuml' ]
               , 'job satisfaction': ['ujbmswrk', 'ujbmsall']
               , 'life satisfaction': ['uatwkhpj', 'ulosat']
               , 'long term employment and job security': [ 'ujbempt'
                                                          , 'ujbmcntr'
                                                          , 'ujbmploj'
                                                          , 'ujbmssec'
                                                          , 'ujbocct'
                                                          , 'ujomsf'
                                                          , 'ujomwf' ]
               , 'recognition': ['ujostat']
               , 'skill-job fit': ['ujomus', 'uskcjed']
               , 'communication with co-workers': ['ujosoc']
               , 'uurnover intentions': ['ujbnewjs']
               , 'variety': ['ujonovil', 'ujomdw', 'ujomrpt', 'ujomvar']
               , 'well being (mental and physical)': ['ujompi', 'ulosatyh']
               , 'work engagement': ['ujbmswrk', 'ujonomfl', 'ujoserve']
               , 'work-life balance': ['ujbmsflx']
               , 'working hours': ['ujbhruc', 'ujbhru', 'ujbmhruw']
               , 'workplace training satisfaction': ['ujbtremp']
               }
ISCO88 ={ 11: "Legislators and senior officials"
        , 12: "Corporate managers"
        , 13: "General managers"
        , 21: "Physical, mathematical and engineering science professionals"
        , 22: "Life science and health professionals"
        , 23: "Teaching professionals"
        , 24: "Other professionals"
        , 31: "Physical and engineering science associate professionals"
        , 32: "Life science and health associate professionals"
        , 33: "Teaching associate professionals"
        , 34: "Other associate professionals"
        , 41: "Office clerks"
        , 42: "Customer services clerks"
        , 51: "Personal and protective services workers"
        , 52: "Models, salespersons and demonstrators"
        , 61: "Market-oriented skilled agricultural and fishery workers"
        , 62: "Subsistence agricultural and fishery workers"
        , 71: "Extraction and building trades workers"
        , 72: "Metal, machinery and related trades workers"
        , 73: "Precision, handicraft, craft printing and related trades workers"
        , 74: "Other craft and related trades workers"
        , 81: "Stationary plant and related operators"
        , 82: "Machine operators and assemblers"
        , 83: "Drivers and mobile plant operators"
        , 91: "Sales and services elementary occupations"
        , 92: "Agricultural, fishery and related labourers"
        , 93: "Labourers in mining, construction, manufacturing and transport" }

# Variables relating to literature of causes/effects of job-satisfaction.
variables = [ var for sublist in list(concepts.values()) for var in sublist ]

# Read the HILDA data one way or another.
if os.path.exists(hilda_spss_path):
    raw, meta = pyreadstat.read_sav(hilda_spss_path)
    hilda = clean(raw)
else:
    with open(hilda_pickle_path, "rb") as f:
        hilda, meta = pickle.load(f)
    with open(raw_pickle_path, "rb") as f:
        raw = pickle.load(f)

labels = { key: meta.column_names_to_labels[key] for key in variables }

# Produce subsets of HILDA based on Fjalar, Brandon or Josh's columns.
hildaf = hilda[fcols.keys()]
hildab = hilda[bcols]
hildaj = hilda[jcols.keys()]
hilda1k = clean(raw.sample(n=1000, random_state=999))
hilda100 = clean(raw.sample(n=100, random_state=999))
hilda25 = clean(raw.sample(n=25, random_state=999))
h25x500 = hilda25.sample(n=500, axis='columns', random_state=11)
# Below subset contains 'ujbmsall':
h100x300 = hilda100.sample(n=300, axis='columns', random_state=99)
h100x300_2 = hilda100.sample(n=300, axis='columns', random_state=9999)

# All the ISCO88 codes with rows in HILDA.
iscosraw = [ int(code) for code in hilda['ujbm682'].unique() ]

# Subsetting.
iscos = [ (isco, hilda[hilda['ujbm682'] == isco].shape[0]) for isco in iscosraw
          if isco not in [1, 34] # Dubious category (1) and +10k rows (34).
        #  if isco % 10 != 0      # Not sure whether to include.
        ]

# ISCO codes with 100 or more rows.
iscover100 = [ isco[0] for isco in iscos if isco[1] > 99 ]

def hilda_by_isco(isco):
    """Get by 2 digit code if `isco` like 21. By 1 digit code if like 2."""
    # Get available iscos - excluding ISCO1 and ISCO34 (dubious or too many).
    iscos = [ int(code) for code in hilda['ujbm682'].unique()
              if int(code) not in [1, 34] ]
    # Inline convenience lookup functions.
    def hilda_by_isco2d(isco_2d): return hilda[hilda['ujbm682'] == isco_2d]
    def first(x): return (x - x % 10) // 10

    # If a 2-digit code is requested.
    if isco > 9:
        return hilda_by_isco2d(isco)
    # If a 1-digit code is requested.
    else:
        return pd.concat([ hilda[hilda['ujbm682'] == code]
                           for code in iscos if first(code) == isco ])

def labeldict(g):
    """Return hildavar to label dictionary for the vertices in `g`."""
    return { col: meta.column_names_to_labels[col] for col in g.nodes() }

def label(g):
    """Return labelled version of HILDA-based graph."""
    return nx.relabel_nodes(g, labeldict(g))


if __name__ == '__main__': pass
