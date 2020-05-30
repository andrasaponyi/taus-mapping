#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 30 10:51:22 2020

@author: andrasaponyi
"""
    
class SentencePair:
    """ Data object for storing information about source-target sentence pairs. """
    
    def __init__(self, source, target):
        self.source = source
        self.target = target
        pass
    
    def add_mapped_vector(self, mapped_vector):
        self.mapped_vector = mapped_vector
        pass
    
    def add_gold_vector(self, gold_vector):
        self.gold_vector = gold_vector
        pass
    
    def add_score(self, score):
        self.score = score
        pass
    
    def to_dict(self):
        
        pair_dict = {
            "source": self.source,
            "target": self.target,
            "mapped_vector": self.mapped_vector,
            "gold_vector": self.gold_vector,
            "score": self.score
            }
    
        return pair_dict
    
    pass