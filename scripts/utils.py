#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 29 18:19:42 2020

@author: andrasaponyi

"""

import random
import pandas as pd
import csv

def shuffle_dict(dictionary):
    
    items = list(dictionary.items())
    random.shuffle(items)
    shuffled = dict(items)
        
    return shuffled

def slice_dict(dictionary, n):
    
    items = list(dictionary.items())
    shorter = dict(items[:n])

    return shorter

def shuffle_and_slice(dictionary, n):

    dictionary = shuffle_dict(dictionary)
    dictionary = slice_dict(dictionary, n)
    
    return dictionary

def tokenize(nlp, sentence):
    
    tokens = list()
    
    doc = nlp(str(sentence))
    for token in doc:
        tokens.append(token.text)
    
    return tokens

def write_results(pairs, scores):
    
    source_sentences = [t[0] for t in pairs]
    target_sentences =[t[1] for t in pairs]
    
    data_tuples = list(zip(source_sentences, target_sentences, scores))
    df = pd.DataFrame(data_tuples, columns=["Mapped", "Gold", "SemScore"])
    df.to_csv("data/results.csv")
    
    pass

def write_pairs_to_file(pairs, scores):
        
    with open("data/results.csv", "w") as csvfile:
        writer = csv.writer(csvfile, delimiter=",")
        writer.writerow(["Source", "Target", "SemScore"])
        for pair in pairs:
            row = [pair.source, pair.target, pair.score]
            writer.writerow(row)

    pass