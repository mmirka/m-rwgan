import numpy as np
import os
import matplotlib.pyplot as plt
from random import randint, randrange
import pickle
import math

from tempfile import mkstemp
from shutil import move, copymode
from os import fdopen, remove

import re
import glob

import routers
from routers import full_DIC_R , complete_DIC_R

complete_DIC_R()  
    
DIC_R = full_DIC_R

def make_X(nbR):
    
    X = np.zeros(nbR)
    
    for i in range(nbR):
        r_type = i%8
        X[i] = r_type
        
    return X

def X2buffers(X):
    bufs = []
    i = 0
    for r_t in X:
        # select type
        r_type = DIC_R[r_t]
        if r_type == "HP":
            r = routers.create_Router_HP(i)
        elif r_type == "MP":
            r = routers.create_Router_MP(i)
        elif r_type == "LP":
            r = routers.create_Router_LP(i)
        elif r_type == "ILP":
        	r = routers.create_Router_ILP(i)
        elif r_type == "ULP":
            r = routers.create_Router_ULP(i)
        elif r_type == "ANE":
            r = routers.create_Router_ANE(i)
        elif r_type == "ANW":
            r = routers.create_Router_ANW(i)
        elif r_type == "ASE":
            r = routers.create_Router_ASE(i)
        elif r_type == "ASW":
            r = routers.create_Router_ASW(i)
        elif r_type == "DNS":
            r = routers.create_Router_DNS(i)
        elif r_type == "DEW":
            r = routers.create_Router_DEW(i) 
        elif r_type == "ANE2":
            r = routers.create_Router_ANE2(i)
        elif r_type == "ANW2":
            r = routers.create_Router_ANW2(i)
        elif r_type == "ASE2":
            r = routers.create_Router_ASE2(i)
        elif r_type == "ASW2":
            r = routers.create_Router_ASW2(i)
        elif r_type == "DNS2":
            r = routers.create_Router_DNS2(i)
        elif r_type == "DEW2":
            r = routers.create_Router_DEW2(i)
        elif r_type == "ANE4":
            r = routers.create_Router_ANE4(i)
        elif r_type == "ANW4":
            r = routers.create_Router_ANW4(i)
        elif r_type == "ASE4":
            r = routers.create_Router_ASE4(i)
        elif r_type == "ASW4":
            r = routers.create_Router_ASW4(i)
        elif r_type == "DNS4":
            r = routers.create_Router_DNS4(i)
        elif r_type == "DEW4":
            r = routers.create_Router_DEW4(i)
        elif r_type == "ANE24":
            r = routers.create_Router_ANE24(i)
        elif r_type == "ANW24":
            r = routers.create_Router_ANW24(i)
        elif r_type == "ASE24":
            r = routers.create_Router_ASE24(i)
        elif r_type == "ASW24":
            r = routers.create_Router_ASW24(i)
        elif r_type == "DNS24":
            r = routers.create_Router_DNS24(i)
        elif r_type == "DEW24":
            r = routers.create_Router_DEW24(i)
        elif r_type == "NE212L4":
            r = routers.create_Router_ANE212L4(i)
        elif r_type == "NW212L4":
            r = routers.create_Router_ANW212L4(i)
        elif r_type == "SE212L4":
            r = routers.create_Router_ASE212L4(i)
        elif r_type == "SW212L4":
            r = routers.create_Router_ASW212L4(i)
        elif r_type == "NS212L4":
            r = routers.create_Router_DNS212L4(i)
        elif r_type == "EW212L4":
            r = routers.create_Router_DEW212L4(i)

        elif r_type == "HOMO":
        	r = routers.create_Router_HOMO(i, r_t-100)
        elif r_type == "local":
        	r = routers.create_Router_local(i, r_t-200)
        elif r_type == "nsew":
        	r = routers.create_Router_nsew(i, r_t-300)
            
        bufs.append(r.buffers)

        i+=1  
    
    return bufs
        
def A2links(A):
    l = len(A)
    links = []
    for i in range(l):
        for j in range(l):
            if A[i,j] == 1:
                if [j,i] in links:
                    pass
                else:
                    links.append([i,j])
    return links

def topo_writer(ned_file):
    base_file = "config_template.ini"
    pattern = "data_file = "
    subst = "data_file = <RATATOSKR_ROOT>/bin/urand/" + NoC_file
    #Create temp file
    fh, abs_path = mkstemp()
    with fdopen(fh,'w') as new_file:
        with open(base_file) as old_file:
            for line in old_file:
                new_file.write(line.replace(pattern, subst))
    #Copy the file permissions from the old file to the new file
    copymode(base_file, abs_path)
    
    move(abs_path, "config.ini")
    
def write_dimension(r,c, file_path):
    pattern = "// Mesh Dim"
    subst = pattern + "\n" + "        rows = " + str(r) + "; \n" + "        columns = " + str(c) + ";"
    
    #Create temp file
    fh, abs_path = mkstemp()
    with fdopen(fh,'w') as new_file:
        with open(file_path) as old_file:
            for line in old_file:
                new_file.write(line.replace(pattern, subst))
    #Copy the file permissions from the old file to the new file
    copymode(file_path, abs_path)
    move(abs_path, file_path)
    
def write_buffers(bufs, file_path):
    pattern = "// bufs :"
    subst = pattern + "\n"
    r_id = 0
    for b_vec in bufs:
        space = "        "
        # North
        s_North = "router[" + str(r_id) + "].bufN = " + str(b_vec[0]) + ";"
        # South
        s_South = "router[" + str(r_id) + "].bufS = " + str(b_vec[1]) + ";"
        # East
        s_East = "router[" + str(r_id) + "].bufE = " + str(b_vec[2]) + ";"
        # West
        s_West = "router[" + str(r_id) + "].bufW = " + str(b_vec[3]) + ";"
        # Local
        s_Local = "router[" + str(r_id) + "].bufL = " + str(b_vec[4]) + ";"
        
        subst += space + s_North + " " + s_South + " " + s_East + " " + s_West + " " + s_Local     
        subst += "\n"
        
        r_id += 1
        
    #Create temp file
    fh, abs_path = mkstemp()
    with fdopen(fh,'w') as new_file:
        with open(file_path) as old_file:
            for line in old_file:
                new_file.write(line.replace(pattern, subst))
    #Copy the file permissions from the old file to the new file
    copymode(file_path, abs_path)
    move(abs_path, file_path)

def write_links(links, file_path):
#ports on routers are 0 = north, 1 = west, 2 = south, 3 = east, 4 = core
# connect south north (all but last row)
#router[r*columns+c].in[2] <--> Link <--> router[(r+1)*columns+c].out[0] if r!=rows-1;
#router[r*columns+c].out[2] <--> Link <--> router[(r+1)*columns+c].in[0] if r!=rows-1;
# connect east west (all but on last column)
#router[r*columns+c].in[3] <--> Link <--> router[r*columns+c+1].out[1] if c!=columns-1;
#router[r*columns+c].out[3] <--> Link <--> router[r*columns+c+1].in[1] if c!=columns-1;
# connect the Cores to port 4
#router[r*columns+c].in[4] <--> Link <--> core[r*columns+c].out;
#router[r*columns+c].out[4] <--> Link <--> core[r*columns+c].in;

    pattern = "// links"
    subst = pattern + "\n"
    space = "        "
    r_id = 0

    for l in links:
        r0 = l[0]
        r1 = l[1]
        row0 = r0//4
        col0 = r0%4
        row1 = r1//4
        col1 = r1%4

        if row0>row1 or row1>row0: #south/north link
            if row0>row1:
                cmd = space + "router["+ str(r1) +"].in[2] <--> Link <--> router["+ str(r0) +"].out[0] ;"
                cmd += "\n"
                cmd += space + "router["+ str(r1) +"].out[2] <--> Link <--> router["+ str(r0) +"].in[0] ; \n"
            else:
                cmd = space +"router["+ str(r0) +"].in[2] <--> Link <--> router["+ str(r1) +"].out[0] ;"
                cmd += "\n"
                cmd += space + "router["+ str(r0) +"].out[2] <--> Link <--> router["+ str(r1) +"].in[0] ; \n"

        else: #east/west link
            if col0>col1:
                cmd = space + "router["+ str(r1) +"].in[3] <--> Link <--> router["+ str(r0) +"].out[1] ;"
                cmd += "\n"
                cmd += space + "router["+ str(r1) +"].out[3] <--> Link <--> router["+ str(r0) +"].in[1] ; \n"
            else:
                cmd = space + "router["+ str(r0) +"].in[3] <--> Link <--> router["+ str(r1) +"].out[1] ;"
                cmd += "\n"
                cmd += space + "router["+ str(r0) +"].out[3] <--> Link <--> router["+ str(r1) +"].in[1] ; \n"

        subst += cmd

    for r in range(16):
        cmd = space + "router["+ str(r) +"].in[4] <--> Link <--> core["+ str(r) +"].out; \n"
        cmd += space + "router["+ str(r) +"].out[4] <--> Link <--> core["+ str(r) +"].in; \n"

        subst += cmd

    #Create temp file
    fh, abs_path = mkstemp()
    with fdopen(fh,'w') as new_file:
        with open(file_path) as old_file:
            for line in old_file:
                new_file.write(line.replace(pattern, subst))
    #Copy the file permissions from the old file to the new file
    copymode(file_path, abs_path)
    move(abs_path, file_path)

def write_NetworkName_noLink(name, file_path):
    pattern = "network myMesh_RouterNSEWL_noLink_template"
    subst = "network " + name

    #Create temp file
    fh, abs_path = mkstemp()
    with fdopen(fh,'w') as new_file:
        with open(file_path) as old_file:
            for line in old_file:
                new_file.write(line.replace(pattern, subst))
    #Copy the file permissions from the old file to the new file
    copymode(file_path, abs_path)
    move(abs_path, file_path)

    
def write_NetworkName(name, file_path):
    pattern = "network myMesh_RouterNSEWL_template"
    subst = "network " + name 
    
    #Create temp file
    fh, abs_path = mkstemp()
    with fdopen(fh,'w') as new_file:
        with open(file_path) as old_file:
            for line in old_file:
                new_file.write(line.replace(pattern, subst))
    #Copy the file permissions from the old file to the new file
    copymode(file_path, abs_path)
    move(abs_path, file_path)

def write_NetworkName_HBw(name, file_path):
    pattern = "network myMesh_RouterNSEWL_template_HBw"
    subst = "network " + name 
    
    #Create temp file
    fh, abs_path = mkstemp()
    with fdopen(fh,'w') as new_file:
        with open(file_path) as old_file:
            for line in old_file:
                new_file.write(line.replace(pattern, subst))
    #Copy the file permissions from the old file to the new file
    copymode(file_path, abs_path)
    move(abs_path, file_path)

    
def ned_writer(c,r,bufs):
    # 1. Hard copy of new topologies.ned
    template_file = "../HNOCS/src/topologies/myMesh_template.ned"
    new_file_path = "../HNOCS/src/topologies/myMesh_new.ned"
    
    #Create temp file
    fh, abs_path = mkstemp()
    with fdopen(fh,'w') as new_file:
        with open(template_file) as old_file:
            for line in old_file:
                new_file.write(line)
    #Copy the file permissions from the old file to the new file
    copymode(template_file, abs_path)
    
    move(abs_path, new_file_path)
    
    # 2. modify topo. dimensions
    if (c != 0) and (r != 0):
        write_dimension(r,c, new_file_path)
        
    # 3. write buffers size
    write_buffers(bufs, new_file_path)
    
    # 4. change network_name
    name = "my_Mesh_toTest"
    write_NetworkName(name, new_file_path)

def ned_writer_holedMesh(c,r,bufs,links):
    # 1. Hard copy of new topologies.ned
    template_file = "../HNOCS/src/topologies/myMesh_template.ned"
    new_file_path = "../HNOCS/src/topologies/myMesh_new.ned"

    #Create temp file
    fh, abs_path = mkstemp()
    with fdopen(fh,'w') as new_file:
        with open(template_file) as old_file:
            for line in old_file:
                new_file.write(line)
    #Copy the file permissions from the old file to the new file
    copymode(template_file, abs_path)

    move(abs_path, new_file_path)

    # 2. modify topo. dimensions
    if (c != 0) and (r != 0):
        write_dimension(r,c, new_file_path)

    # 3. write buffers size
    write_buffers(bufs, new_file_path)

    # 4. write links
    write_links(links, new_file_path)

    # 5. change network_name
    name = "my_Mesh_toTest"
    write_NetworkName(name, new_file_path)

def ned_writer_3x4_multi(simuID,c,r,bufs):
    # 1. Hard copy of new topologies.ned
    template_file = "../HNOCS/src/topologies/myMesh_template.ned"
    new_file_path = "../HNOCS/src/topologies/myMesh_simu"+str(simuID)+".ned"
    new_temp_file_path = "<WORK_ROOT>/myMesh_simu"+str(simuID)+".ned"

    #Create temp file
    #fh, abs_path = mkstemp()
    with open(new_temp_file_path,'w') as new_file:
        with open(template_file) as old_file:
            for line in old_file:
                new_file.write(line)
    #Copy the file permissions from the old file to the new file
    copymode(template_file, new_temp_file_path)

    #move(abs_path, new_file_path)

    # 2. modify topo. dimensions
    if (c != 0) and (r != 0):
        write_dimension(r,c, new_temp_file_path)

    # 3. write buffers size
    write_buffers(bufs, new_temp_file_path)

    # 4. change network_name
    name = "my_Mesh_simu"+str(simuID)
    write_NetworkName(name, new_temp_file_path)

    move(new_temp_file_path, new_file_path)



def ned_writer_multi(simuID,c,r,bufs):
    # 1. Hard copy of new topologies.ned
    template_file = "../HNOCS/src/topologies/myMesh_template.ned"
    new_file_path = "../HNOCS/src/topologies/myMesh_simu"+str(simuID)+".ned"
    new_temp_file_path = "<WORK_ROOT>/myMesh_simu"+str(simuID)+".ned"

    #Create temp file
    #fh, abs_path = mkstemp()
    with open(new_temp_file_path,'w') as new_file:
        with open(template_file) as old_file:
            for line in old_file:
                new_file.write(line)
    #Copy the file permissions from the old file to the new file
    copymode(template_file, new_temp_file_path)

    #move(abs_path, new_file_path)

    # 2. modify topo. dimensions
    if (c != 0) and (r != 0):
        write_dimension(r,c, new_temp_file_path)

    # 3. write buffers size
    write_buffers(bufs, new_temp_file_path)

    # 4. change network_name
    name = "my_Mesh_simu"+str(simuID)
    write_NetworkName(name, new_temp_file_path)

    move(new_temp_file_path, new_file_path)

def ned_writer_multi_holedMesh(simuID,c,r,bufs, links):
    # 1. Hard copy of new topologies.ned
    template_file = "../HNOCS/src/topologies/myMesh_template_nolink.ned"
    new_file_path = "../HNOCS/src/topologies/myMesh_simu"+str(simuID)+".ned"
    new_temp_file_path = "<WORK_ROOT>/myMesh_simu"+str(simuID)+".ned"

    #Create temp file
    #fh, abs_path = mkstemp()
    with open(new_temp_file_path,'w') as new_file:
        with open(template_file) as old_file:
            for line in old_file:
                new_file.write(line)
    #Copy the file permissions from the old file to the new file
    copymode(template_file, new_temp_file_path)

    #move(abs_path, new_file_path)

    # 2. modify topo. dimensions
    if (c != 0) and (r != 0):
        write_dimension(r,c, new_temp_file_path)

    # 3. write buffers size
    write_buffers(bufs, new_temp_file_path)

    # 4. write links
    write_links(links, new_temp_file_path)

    # 5. change network_name
    name = "my_Mesh_simu"+str(simuID)
    write_NetworkName_noLink(name, new_temp_file_path)

    move(new_temp_file_path, new_file_path)


def ned_writer_multi_HBw(simuID,c,r,bufs):
    # 1. Hard copy of new topologies.ned
    template_file = "<HNOCS_ROOT>/src/topologies/myMesh_template_HBw.ned"
    new_file_path = "<HNOCS_ROOT>/src/topologies/myMesh_simu"+str(simuID)+".ned"
    
    #Create temp file
    fh, abs_path = mkstemp()
    with fdopen(fh,'w') as new_file:
        with open(template_file) as old_file:
            for line in old_file:
                new_file.write(line)
    #Copy the file permissions from the old file to the new file
    copymode(template_file, abs_path)
    
    move(abs_path, new_file_path)
    
    # 2. modify topo. dimensions
    if (c != 0) and (r != 0):
        write_dimension(r,c, new_file_path)
        
    # 3. write buffers size
    write_buffers(bufs, new_file_path)
    
    # 4. change network_name
    name = "my_Mesh_simu"+str(simuID)
    write_NetworkName_HBw(name, new_file_path)

def write(X): # main funtion to include in other scripts

    c = int(math.sqrt(len(X)))
    r = c
    ## get bufers list
    bufs = X2buffers(X)
    
    ## write .ned
    ned_writer(c,r,bufs)

def write_simuID_holedMesh(X, A, simuID): # main funtion to include in other scripts

    c = int(math.sqrt(len(X)))
    r = c
    ## get bufers list
    bufs = X2buffers(X)
    links = A2links(A)

    ## write .ned
    ned_writer_multi_holedMesh(simuID, c, r, bufs, links)

def write_simuID(X, simuID): # main funtion to include in other scripts

    c = int(math.sqrt(len(X)))
    r = c
    ## get bufers list
    bufs = X2buffers(X)
    
    ## write .ned
    ned_writer_multi(simuID,c,r,bufs)

def write_simuID_HBw(X, simuID): # main funtion to include in other scripts

    c = int(math.sqrt(len(X)))
    r = c
    ## get bufers list
    bufs = X2buffers(X)
    
    ## write .ned
    ned_writer_multi_HBw(simuID,c,r,bufs)

def write_simuID_3x4(X, simuID): # main funtion to include in other scripts

    c = 4
    r = 3
    ## get bufers list
    bufs = X2buffers(X)

    ## write .ned
    ned_writer_multi(simuID,c,r,bufs)

    
def main(): # test function script
    
    c = 4
    r = 4
    ## create X matrix 
    nbR = c*r
    X = make_X(nbR)
    
    ## get bufers list
    bufs = X2buffers(X)
    
    ## write .ned
   # ned_writer(c,r,bufs)

    write(X)
    
main()
