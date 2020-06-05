#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 29 18:19:42 2020

@author: andrasaponyi

"""

import random
import pandas as pd
import csv

def pairs_from_df(infile):
    
    df = pd.read_csv(infile, index_col=0)
    sources = list(df.Source)
    targets = list(df.Target)
    pairs = list(zip(sources, targets))
    
    return pairs

def shuffle_data(data_tuples):
    
    random.shuffle(data_tuples)
    shuffled = data_tuples
        
    return shuffled

def slice_data(data_tuples, n):
    
    shorter = data_tuples[:n]

    return shorter

def shuffle_and_slice(data_tuples, n):

    data_tuples = shuffle_data(data_tuples)
    data_tuples = slice_data(data_tuples, n)
    
    return data_tuples

def tokenize(nlp, sentence):
    
    tokens = list()
    
    doc = nlp(str(sentence))
    for token in doc:
        tokens.append(token.text)
    
    return tokens

def write_results(pairs):
    """ This is a very convulated way of doing things - fix later. """
    
    source_sentences = [pair.source for pair in pairs]
    target_sentences = [pair.target for pair in pairs]
    scores = [pair.score for pair in pairs]
    
    data_tuples = list(zip(source_sentences, target_sentences, scores))
    df = pd.DataFrame(data_tuples, columns=["Source", "Target", "SemScore"])
    df.to_csv("data/dataframe_with_semscores.csv")
    
    pass