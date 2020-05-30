#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 10:51:00 2020

@author: andrasaponyi

Obtain a linear or orthogonal transformation between two monolingual vectors spaces.
Creates file "transformation_matrix.csv" in "data" directory.
Run from main directory (taus-mapping)!
"""

import numpy as np
import scipy as sp
import json
from gensim.models import KeyedVectors
import argparse
parser = argparse.ArgumentParser()
    
def seed_to_lists(model_source, model_target, seed_dict):
    """
    Remove words from the seed dictionary that are missing from either model 
    Return source and target words as separate lists.
    This returns a dictionary that is considerably shorter than the original, bvt
    out-of-vocabulary words are treated.
    """
    
    source_words = list()
    target_words = list()
        
    for source_word, target_word in seed_dict.items():
        if source_word in model_source and target_word in model_target:
            source_words.append(source_word)
            target_words.append(target_word)
                        
    return source_words, target_words

def get_vectors(model, words):
    """ Retrieve word vectors from a given word2vec model. """
    
    vectors = list()
    
    for word in words:
        vector = model[word]
        vectors.append(vector)
        
    return vectors

def find_translation_matrix(model_source, model_target, seed_dictionary, method):
    """
    Given matrices source and target, find a matrix that most closely maps source to target.
    LSTSQ: least squares solution.
    ORTH: solution to the orthogonal Procrustes problem. Uses SVD.
    """
    
    source_words, target_words = seed_to_lists(model_source, model_target, seed_dictionary)
    
    source_matrix = get_vectors(model_source, source_words)
    target_matrix = get_vectors(model_target, target_words)
    
    if method == "lstsq":
        translation_matrix = np.linalg.lstsq(source_matrix, target_matrix, rcond=None)[0].T
    elif method == "orth":
        translation_matrix = sp.linalg.orthogonal_procrustes(source_matrix, target_matrix)[0].T
    else:
        raise ValueError("Invalid method name.")
    
    return translation_matrix

def main(method):
    
    # load models trained using gensim implementation of word2vec
    # model_source = "vectors/source_vectors.bin"
    # model_target = "vectors/target_vectors.bin"
    model_source = "vectors/taus_en_300.bin"
    model_target = "vectors/taus_fr_300.bin"
    model_source = KeyedVectors.load_word2vec_format(model_source, binary=True)
    model_target = KeyedVectors.load_word2vec_format(model_target, binary=True)

    # list of word pairs to train translation matrix as json
    seed_dictionary_file = "data/seed_dictionary.json"
    with open(seed_dictionary_file, "r") as json_file:
        seed_dictionary = json.load(json_file)
        
    # find translaton matrix
    translation_matrix = find_translation_matrix(model_source, model_target, seed_dictionary, method)
    
    # Save learned translaton matrix to file.
    np.savetxt("data/transformation_matrix.csv", translation_matrix, delimiter=",")
    
if __name__ == "__main__":
        
    parser.add_argument("-mtd", "--method")
    args = parser.parse_args()
    main(args.method)    