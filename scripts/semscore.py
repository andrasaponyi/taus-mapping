#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 30 11:49:26 2020

@author: andrasaponyi

Generate semantic similarity scores given a list of sentence pairs and a mapping method.
"""

import mapping
import calculate_sentence_score
import argparse
parser = argparse.ArgumentParser()

def main(method, n=None):
    
    mapping.main(method)
    calculate_sentence_score.main(n)
    
    pass

if __name__ == "__main__":
    
    parser.add_argument("-mtd", "--method")
    parser.add_argument("-n", "--n")
    args = parser.parse_args()
    main(args.method, args.n)