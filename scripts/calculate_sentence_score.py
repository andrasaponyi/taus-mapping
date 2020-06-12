#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  7 09:27:59 2020

@author: andrasaponyi

Calculate the cosine similarity between the mean vectors of a pair of sentences.
for dev: (= sentence_score_new.py)
Run from main directory (taus-mapping)!
"""

import numpy as np
from gensim.models import KeyedVectors
import spacy
from sklearn.metrics.pairwise import cosine_similarity
from utils import shuffle_and_slice, tokenize
from utils import write_results, pairs_from_df
from classes import SentencePair
import argparse
parser = argparse.ArgumentParser()

    
def calculate_sentence_vector(model, tokens):
    """
    A sentence vector is defined as the element-wise mean of all of its word vectors.
    Out-of-vocabulary words are ignored.
    """
    
    vectors = list()
    for token in tokens:
        try:
            vector = model[token]
            vectors.append(vector)
        except:
            pass

    sentence_vector = np.mean(vectors, axis=0)
          
    return sentence_vector

def calculate_gold_vectors(model_target, nlp_target, pairs):
    """ Calculate sentence vectors for target language phrases. """
    
    gold_vectors = list()
    
    for source, target in pairs.items():
        tokens = tokenize(nlp_target, target)
        sent_vector = calculate_sentence_vector(model_target, tokens)
        gold_vectors.append(sent_vector)
    
    return gold_vectors

def map_tokens_to_sent_vector(model_source, tm, source_tokens):
    """
    Map all words a tokenized sentence into a target language space.
    Calculate and return a sentence vector.
    """
    
    mapped_vectors = list()
    
    for token in source_tokens:
        try:
            # Currently not looking for nearest neighbor.
            pivot = tm.dot(model_source[token])
            mapped_vectors.append(pivot)
        except:
            pass
            
    # Setting dtype to float32 to match gold format.
    sentence_vector = np.mean(mapped_vectors, axis=0, dtype="float32")
    
    return sentence_vector

def calculate_sentence_vectors(model_source, model_target, tm, pair_objects):
    """ 
    Given a dictionary of sentence pairs,
    calculate sentence vectors from both source and target.
    Return two lists of sentence vectors.
    """
    
    nlp_source = spacy.load("en_core_web_md")
    nlp_target = spacy.load("fr_core_news_md")
    
    pairs = list()
    
    # for source, target in pairs.items():
    for pair in pair_objects:
        
        # tokenize sentence
        source_tokens = tokenize(nlp_source, pair.source)
        target_tokens = tokenize(nlp_target, pair.target)
                
        # obtained a sentence vector for the source sentence
        # first, map source word vectors into the target space using a transformation matrix
        # second, calculate a sentence vector from the mapped word vectors
        mapped_sent_vector = map_tokens_to_sent_vector(model_source, tm, source_tokens)
        pair.add_mapped_vector(mapped_sent_vector)
                
        # calculate a sentence vector from target word vectors
        gold_sent_vector = calculate_sentence_vector(model_target, target_tokens)
        pair.add_gold_vector(gold_sent_vector)
        
        pairs.append(pair)
            
    return pairs

def compare_sentence_vectors(pairs):
    """
    Calculate cosine similarity between
    mapped source sentence vectors and target sentence vectors.
    Return a list of similarity scores.
    """
    
    scores = list()
    
    for pair in pairs:
        try:
            sim = cosine_similarity([pair.mapped_vector], [pair.gold_vector])[0][0]
            sim = abs(sim)
            scores.append(sim)
        except:
            scores.append(0.5)
            pass
    
    return scores

def add_scores_to_pairs(pairs, scores):
    
    new_pairs = list()
    
    tuples = list(zip(pairs, scores))
    for t in tuples:
        t[0].add_score(t[1])
        new_pairs.append(t[0])
    
    return new_pairs

def main(n=None):
    
    if n:
        n = int(n)
    
    # Load word2vec models.
    model_source = "vectors/source_vectors.bin"
    model_target = "vectors/target_vectors.bin"
    # model_source = "word2vec/taus_en_300.bin"
    # model_target = "word2vec/taus_fr_300.bin"
    model_source = KeyedVectors.load_word2vec_format(model_source, binary=True)
    model_target = KeyedVectors.load_word2vec_format(model_target, binary=True)
    
    # Load translation matrix from file.
    tm = np.loadtxt("data/transformation_matrix.csv", delimiter=",")
    
    # Pairs: a CSV file containing source and target sentence pairs.
    # pairsfile = "data/pairs.csv"
    pairsfile = "data/sample_pairs.csv"
    if n:
        # Shorten data to n pairs to make testing faster.
        pairs = pairs_from_df(pairsfile)
        pairs = shuffle_and_slice(pairs, n)
    else:
        pairs = pairs_from_df(pairsfile)
            
    pair_objects = list()
    for p in pairs:
        # define pair objects
        pair = SentencePair(p[0], p[1])
        pair_objects.append(pair)
    
    pairs = calculate_sentence_vectors(model_source, model_target, tm, pair_objects)
    scores = compare_sentence_vectors(pairs)
    pairs = add_scores_to_pairs(pairs, scores)
    
    average = np.mean(scores)
    print("Scores average: ", average)
    
    write_results(pairs)
    
if __name__ == "__main__":
    
    parser.add_argument("-n", "--n")
    args = parser.parse_args()
    main(args.n)