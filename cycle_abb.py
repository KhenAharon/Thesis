#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 15:37:20 2019

@author: simsim
"""

from system import *

# example permitted from Doron's paper

def create(model):
    plant = process("plant",["p0"],[],[],"p0")
    environment = process("environment",["e0","e1","e2"],[],[],"e0")
    
    pve = plant_environment("syst",plant,environment,model = model)
    
    pve.add_transition("a",["plant","environment"],[["p0"],["e0"]],[["p0"],["e1"]])
    pve.add_transition("b",["plant","environment"],[["p0"],["e1","e2"]],[["p0"],["e2","e0"]])

    pve.create_RNN()
    pve.reinitialize()
    return pve