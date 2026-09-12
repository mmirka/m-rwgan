############################################################
# This file contains various denominations of routers architectures.
# All are defined in the full_DIC_R disctionary, where the name of the router is linked to a code.
# Each type of router has a creator function.
# Both dictionnary and functions are used in the ned_writer.py script to write simulations files.
#
# The main family of routeurs used is the homogeneous routers (i.e. same buffer size for all ports), define by the name "HOMO" and created with the function create_Router_HOMO(ID,size_buf), where size_buf defines the size of all buffers.
#
# We mainly used the codes 112, 104, 102 for homogeneous routers with buffers of size 12, 4, 2 flits.
# They are referred in our work as Big, Medium and Small routers.

###########################################################

import numpy as np

full_DIC_R = {0 : "HP", 1 : "LP", 2 : "ANE", 3 : "ANW", 4 : "ASE", 5 : "ASW", 6 : "DNS", 7 : "DEW", 8:"MP", 9:"ULP", 10:"ILP", 12 : "ANE2", 13 : "ANW2", 14 : "ASE2", 15 : "ASW2", 16 : "DNS2", 17 : "DEW2", 22 : "ANE4", 23 : "ANW4", 24 : "ASE4", 25 : "ASW4", 26 : "DNS4", 27 : "DEW4", 32 : "ANE24", 33 : "ANW24", 34 : "ASE24", 35 : "ASW24", 36 : "DNS24", 37 : "DEW24", 100:"HOMO", 200:"local", 300:"nsew", 42 : "NE212L4", 43 : "NW212L4", 44 : "SE212L4", 45 : "SW212L4", 46 : "NS212L4", 47 : "EW212L4"}


def complete_DIC_R():
    for i in range(21):
        full_DIC_R[100+i] = "HOMO"
        full_DIC_R[200+i] = "local"
        full_DIC_R[300+i] = "nsew"


class Router:
    def __init__(self, ID, bufs): 
        self.id = ID
        self.buffers = bufs # bufs = [N, S, E, W, L]

def create_Router_HOMO(ID,size_buf): # high saturation point
    b_N = size_buf
    b_S = size_buf
    b_E = size_buf
    b_W = size_buf
    b_L = size_buf
    bufs = [b_N, b_S, b_E, b_W, b_L]
    
    R = Router(ID, bufs)
    
    return R

def create_Router_local(ID,size_buf): # high saturation point
    b_N = 4
    b_S = 4
    b_E = 4
    b_W = 4
    b_L = size_buf
    bufs = [b_N, b_S, b_E, b_W, b_L]
    
    R = Router(ID, bufs)
    
    return R

def create_Router_nsew(ID,size_buf): # high saturation point
    b_N = size_buf
    b_S = size_buf
    b_E = size_buf
    b_W = size_buf
    b_L = 4
    bufs = [b_N, b_S, b_E, b_W, b_L]
    
    R = Router(ID, bufs)
    
    return R


def create_Router_HP(ID): # high saturation point
    b_N = 12
    b_S = 12
    b_E = 12
    b_W = 12
    b_L = 12
    bufs = [b_N, b_S, b_E, b_W, b_L]
    
    R = Router(ID, bufs)
    
    return R

def create_Router_MP(ID): # medium saturation point 
    b_N = 8
    b_S = 8
    b_E = 8
    b_W = 8
    b_L = 8
    bufs = [b_N, b_S, b_E, b_W, b_L]
    
    R = Router(ID, bufs)
    
    return R

def create_Router_LP(ID): # low saturation point 
    b_N = 4
    b_S = 4
    b_E = 4
    b_W = 4
    b_L = 4
    bufs = [b_N, b_S, b_E, b_W, b_L]
    
    R = Router(ID, bufs)
    
    return R

def create_Router_ILP(ID):
    b_N = 2
    b_S = 2
    b_E = 2
    b_W = 2
    b_L = 2
    bufs = [b_N, b_S, b_E, b_W, b_L]
    
    R = Router(ID, bufs)
    
    return R

def create_Router_ULP(ID): # low saturation point 
    b_N = 1
    b_S = 1
    b_E = 1
    b_W = 1
    b_L = 1
    bufs = [b_N, b_S, b_E, b_W, b_L]
    
    R = Router(ID, bufs)
    
    return R

def create_Router_ANE(ID): # angle North-East
    b_N = 1
    b_S = 8
    b_E = 1
    b_W = 8
    b_L = 8
    bufs = [b_N, b_S, b_E, b_W, b_L]
    
    R = Router(ID, bufs)
    
    return R

def create_Router_ANW(ID): # angle North-West
    b_N = 1
    b_S = 8
    b_E = 8
    b_W = 1
    b_L = 8
    bufs = [b_N, b_S, b_E, b_W, b_L]
    
    R = Router(ID, bufs)
    
    return R

def create_Router_ASE(ID): # angle South-East
    b_N = 8
    b_S = 1
    b_E = 1
    b_W = 8
    b_L = 8
    bufs = [b_N, b_S, b_E, b_W, b_L]
    
    R = Router(ID, bufs)
    
    return R

def create_Router_ASW(ID): # angle South-West
    b_N = 8
    b_S = 1 
    b_E = 8
    b_W = 1
    b_L = 8
    bufs = [b_N, b_S, b_E, b_W, b_L]
    
    R = Router(ID, bufs)
    
    return R
        
def create_Router_DNS(ID): # direction North-South
    b_N = 8
    b_S = 8
    b_E = 1
    b_W = 1
    b_L = 8
    bufs = [b_N, b_S, b_E, b_W, b_L]
    
    R = Router(ID, bufs)
    
    return R  

def create_Router_DEW(ID): # direction East-West
    b_N = 1
    b_S = 1
    b_E = 8
    b_W = 8
    b_L = 8
    bufs = [b_N, b_S, b_E, b_W, b_L]
    
    R = Router(ID, bufs)
    
    return R


def create_Router_ANE2(ID): # angle North-East
    b_N = 2
    b_S = 8
    b_E = 2
    b_W = 8
    b_L = 8
    bufs = [b_N, b_S, b_E, b_W, b_L]
    
    R = Router(ID, bufs)
    
    return R

def create_Router_ANW2(ID): # angle North-West
    b_N = 2
    b_S = 8
    b_E = 8
    b_W = 2
    b_L = 8
    bufs = [b_N, b_S, b_E, b_W, b_L]
    
    R = Router(ID, bufs)
    
    return R

def create_Router_ASE2(ID): # angle South-East
    b_N = 8
    b_S = 2
    b_E = 2
    b_W = 8
    b_L = 8
    bufs = [b_N, b_S, b_E, b_W, b_L]
    
    R = Router(ID, bufs)
    
    return R

def create_Router_ASW2(ID): # angle South-West
    b_N = 8
    b_S = 2 
    b_E = 8
    b_W = 2
    b_L = 8
    bufs = [b_N, b_S, b_E, b_W, b_L]
    
    R = Router(ID, bufs)
    
    return R
        
def create_Router_DNS2(ID): # direction North-South
    b_N = 8
    b_S = 8
    b_E = 2
    b_W = 2
    b_L = 8
    bufs = [b_N, b_S, b_E, b_W, b_L]
    
    R = Router(ID, bufs)
    
    return R  

def create_Router_DEW2(ID): # direction East-West
    b_N = 2
    b_S = 2
    b_E = 8
    b_W = 8
    b_L = 8
    bufs = [b_N, b_S, b_E, b_W, b_L]
    
    R = Router(ID, bufs)
    
    return R

def create_Router_ANE4(ID): # angle North-East
    b_N = 4
    b_S = 8
    b_E = 4
    b_W = 8
    b_L = 8
    bufs = [b_N, b_S, b_E, b_W, b_L]
    
    R = Router(ID, bufs)
    
    return R

def create_Router_ANW4(ID): # angle North-West
    b_N = 4
    b_S = 8
    b_E = 8
    b_W = 4
    b_L = 8
    bufs = [b_N, b_S, b_E, b_W, b_L]
    
    R = Router(ID, bufs)
    
    return R

def create_Router_ASE4(ID): # angle South-East
    b_N = 8
    b_S = 4
    b_E = 4
    b_W = 8
    b_L = 8
    bufs = [b_N, b_S, b_E, b_W, b_L]
    
    R = Router(ID, bufs)
    
    return R

def create_Router_ASW4(ID): # angle South-West
    b_N = 8
    b_S = 4 
    b_E = 8
    b_W = 4
    b_L = 8
    bufs = [b_N, b_S, b_E, b_W, b_L]
    
    R = Router(ID, bufs)
    
    return R
        
def create_Router_DNS4(ID): # direction North-South
    b_N = 8
    b_S = 8
    b_E = 4
    b_W = 4
    b_L = 8
    bufs = [b_N, b_S, b_E, b_W, b_L]
    
    R = Router(ID, bufs)
    
    return R  

def create_Router_DEW4(ID): # direction East-West
    b_N = 4
    b_S = 4
    b_E = 8
    b_W = 8
    b_L = 8
    bufs = [b_N, b_S, b_E, b_W, b_L]
    
    R = Router(ID, bufs)
    
    return R

def create_Router_ANE24(ID): # angle North-East
    b_N = 2
    b_S = 4
    b_E = 2
    b_W = 4
    b_L = 4
    bufs = [b_N, b_S, b_E, b_W, b_L]
    
    R = Router(ID, bufs)
    
    return R

def create_Router_ANW24(ID): # angle North-West
    b_N = 2
    b_S = 4
    b_E = 4
    b_W = 2
    b_L = 4
    bufs = [b_N, b_S, b_E, b_W, b_L]
    
    R = Router(ID, bufs)
    
    return R

def create_Router_ASE24(ID): # angle South-East
    b_N = 4
    b_S = 2
    b_E = 2
    b_W = 4
    b_L = 4
    bufs = [b_N, b_S, b_E, b_W, b_L]
    
    R = Router(ID, bufs)
    
    return R

def create_Router_ASW24(ID): # angle South-West
    b_N = 4
    b_S = 2 
    b_E = 4
    b_W = 2
    b_L = 4
    bufs = [b_N, b_S, b_E, b_W, b_L]
    
    R = Router(ID, bufs)
    
    return R
        
def create_Router_DNS24(ID): # direction North-South
    b_N = 4
    b_S = 4
    b_E = 2
    b_W = 2
    b_L = 4
    bufs = [b_N, b_S, b_E, b_W, b_L]
    
    R = Router(ID, bufs)
    
    return R  

def create_Router_DEW24(ID): # direction East-West
    b_N = 2
    b_S = 2
    b_E = 4
    b_W = 4
    b_L = 4
    bufs = [b_N, b_S, b_E, b_W, b_L]
    
    R = Router(ID, bufs)
    
    return R

def create_Router_ANE212L4(ID): # angle North-East
    b_N = 12
    b_S = 2
    b_E = 12
    b_W = 2
    b_L = 4
    bufs = [b_N, b_S, b_E, b_W, b_L]

    R = Router(ID, bufs)

    return R

def create_Router_ANW212L4(ID): # angle North-West
    b_N = 12
    b_S = 2
    b_E = 2
    b_W = 12
    b_L = 4
    bufs = [b_N, b_S, b_E, b_W, b_L]

    R = Router(ID, bufs)

    return R

def create_Router_ASE212L4(ID): # angle South-East
    b_N = 2
    b_S = 12
    b_E = 12
    b_W = 2
    b_L = 4
    bufs = [b_N, b_S, b_E, b_W, b_L]

    R = Router(ID, bufs)

    return R

def create_Router_ASW212L4(ID): # angle South-West
    b_N = 2
    b_S = 12
    b_E = 2
    b_W = 12
    b_L = 4
    bufs = [b_N, b_S, b_E, b_W, b_L]

    R = Router(ID, bufs)

    return R

def create_Router_DNS212L4(ID): # direction North-South
    b_N = 12
    b_S = 12
    b_E = 2
    b_W = 2
    b_L = 4
    bufs = [b_N, b_S, b_E, b_W, b_L]

    R = Router(ID, bufs)

    return R

def create_Router_DEW212L4(ID): # direction East-West
    b_N = 2
    b_S = 2
    b_E = 12
    b_W = 12
    b_L = 4
    bufs = [b_N, b_S, b_E, b_W, b_L]

    R = Router(ID, bufs)

    return R


