############################################################################################
# 		Script to create a dataset of 10k data.
############################################################################################
# A data consists in a 3x4 NoC alongside its performances.
#
# NoCs are described with 2 matrices:
#	. A = adjacency matrix, same for all as we only consider regular meshes
#	. X = features matrix, comprises the classe of each routeur, from a set a 3 classes
#	
# We randomly create a set of 10k X matrices (i.e. NoCs), using the function "generate_random_Xset_3classes()".
#
# We compute the performances with the function "generate_dataset_fromX_dynamic()"
#
############################################################################################


import numpy as np
import os
import matplotlib.pyplot as plt
from random import randint, randrange
import random
import pickle
import math

from tempfile import mkstemp
from shutil import move, copymode
from os import fdopen, remove

import re
import glob

import ned_writer	
import routers

from routers import full_DIC_R as DIC_R

import copy

import subprocess

from multiprocessing import Process, Pool
import threading

import time

#################### to adapt #########################
# path to the simulator directory
hnocs_path = "../HNOCS/" 
#######################################################

######################
# 8x8 Adjacency Matrix
######################

size_c = 8 # side length of the mesh
nb_routers = size_c*size_c


A_mesh8x8 = np.zeros((nb_routers, nb_routers)) # if A[i,j] = 1 --> connection between routers i and j

## Create a size_cXsize_c mesh adjacency matrix

for r1 in range(nb_routers): # for all routers r1
    for r2 in range(nb_routers): # for all routers r2 --> for all pair of routers
        if r1 == r2: # we do not include the connection to itself
            pass
        else:
            # get positions of r1 and r2
            pos_x1 = r1%size_c
            pos_y1 = r1//size_c
            
            pos_x2 = r2%size_c
            pos_y2 = r2//size_c
            
            if pos_x1 == pos_x2: # if same column
                if abs(pos_y2 - pos_y1)==1: # if on consecutive row --> connection
                    # bidirectional connection: r1 --> r2 and r2 --> r1
                    A_mesh8x8[r1,r2] = 1 
                    A_mesh8x8[r2,r1] = 1
                else:
                    pass
            elif pos_y1 == pos_y2: # if same row
                if abs(pos_x2 - pos_x1)==1: # if on consecutive column --> connection
                    # bidirectional connection: r1 --> r2 and r2 --> r1
                    A_mesh8x8[r1,r2] = 1
                    A_mesh8x8[r2,r1] = 1
                else:
                    pass
            else:
                pass
print(A_mesh8x8)



######################
# 3x4 Adjacency Matrix
######################


A_mesh3x4 = np.zeros((12, 12)) # if A[i,j] = 1 --> connection between routers i and j

## Create a size_cXsize_c mesh adjacency matrix

for r1 in range(12): # for all routers r1
    for r2 in range(12): # for all routers r2 --> for all pair of routers
        if r1 == r2: # we do not include the connection to itself
            pass
        else:
            # get positions of r1 and r2
            pos_x1 = r1%4
            pos_y1 = r1//4

            pos_x2 = r2%4
            pos_y2 = r2//4

            if pos_x1 == pos_x2: # if same column
                if abs(pos_y2 - pos_y1)==1: # if on consecutive row --> connection
                    # bidirectional connection: r1 --> r2 and r2 --> r1
                    A_mesh3x4[r1,r2] = 1
                    A_mesh3x4[r2,r1] = 1
                else:
                    pass
            elif pos_y1 == pos_y2: # if same row
                if abs(pos_x2 - pos_x1)==1: # if on consecutive column --> connection
                    # bidirectional connection: r1 --> r2 and r2 --> r1
                    A_mesh3x4[r1,r2] = 1
                    A_mesh3x4[r2,r1] = 1
                else:
                    pass
            else:
                pass
print(A_mesh3x4)




# Data class

class Data_AX:
    def __init__(self, A, X, latency, total_power, powers, total_area, areas, total_jouls): 
        self.A = A
        self.X = X
        self.latency = latency
        self.total_power = total_power
        self.powers = powers
        self.total_area = total_area
        self.areas = areas
        self.total_jouls = total_jouls


# first version of the saving function
# collect the simulation results to produce the final dataset
def save_dataset(A, X_list, folder_path):
	dataset = []
	filename = str(folder_path) +"dataset"
	for i in range(len(X_list)):
		A = A
		X_mat = X_list[i]
		## latency:
		lat_file = str(folder_path) + "sim_"+str(i)+"/latencies/__end2endLatency.stat"
		in_file = open(lat_file, 'r')
		X=[]
		Y=[]
		while(1):
			#parse input file
			line = in_file.readline()
			if (len(line) == 0):
				break
			fields = line.split(";")
			X.append(float(fields[0]))
			Y.append(float(fields[1]))
			if(float(fields[0]) >= 0.7):
				break
		latency = [X,Y]
		print(latency)

        ## power
        ##- global
		print("\n POWER \n")
		pow_file = str(folder_path) + "sim_"+str(i)+"/powers/__globalPower.stat"
		in_file = open(pow_file, 'r')
		X1=[]
		Y1=[]
		while(1):
			#parse input file
			line = in_file.readline()
			if (len(line) == 0):
				break
			fields = line.split(";")
			X1.append(float(fields[0]))
			Y1.append(float(fields[1]))
			if(float(fields[0]) >= 0.7):
				break
		total_power = [X1,Y1]
		print(total_power)

		###- individual
		powers = []
		for j in range(len(A)):
			pow_file = str(folder_path) + "sim_"+str(i)+"/powers/__router_"+str(j)+"_Power.stat"
			in_file = open(pow_file, 'r')
			X2=[]
			Y2=[]
			while(1):
				#parse input file
				line = in_file.readline()
				if (len(line) == 0):
					break
				fields = line.split(";")
				X2.append(float(fields[0]))
				Y2.append(float(fields[1]))
				if(float(fields[0]) >= 0.7):
					break
			r_power = [X2,Y2]
			powers.append(copy.deepcopy(r_power))
			print(r_power)
		print(powers)

		### area
		print("\n AERA \n")
		area_file = str(folder_path) + "sim_"+str(i)+"/area/area.area"
		print(area_file)
		in_file = open(area_file, 'r')
		areas=[]
		cpt_line = 0
		while(1):
			#parse input file
			line = in_file.readline()
			print(line)
			if cpt_line == 0:
				line = in_file.readline()
				print(line)

			if (len(line) == 0):
				break
			fields = line.split(";")
			areas.append(float(fields[4]))
			if(float(fields[0]) >= 15):
				break
			cpt_line += 1
		total_area = sum(areas)
		print(areas)
		print(total_area)

		### jouls/bytes
		print("\n JOULSpFILTS \n")
		jouls_file = str(folder_path) + "sim_"+str(i)+"/jouls/__efficiency.stat"
		in_file = open(jouls_file, 'r')
		Xj=[]
		Yj=[]
		while(1):
			#parse input file
			line = in_file.readline()
			if (len(line) == 0):
				break
			fields = line.split(";")
			Xj.append(float(fields[0]))
			Yj.append(float(fields[1]))
			if(float(fields[0]) >= 0.7):
				break
		total_jouls = [Xj,Yj]
		print(total_jouls)

		### add to dataset
		data = Data_AX(A, X_mat, latency, total_power, powers, total_area, areas, total_jouls)
		print(data)
		dataset.append(data)
		### Clean repertory


	with open(filename, 'wb') as f:
		pickle.dump(dataset, f)

	print(dataset)




# Randomly generate a set of nb_samples NoCs (as X matrices), of size rxc
# Each element is of the set of classes [112,104,102]
def generate_random_Xset_3classes(nb_samples, c, r):
        x_size = c*r
        classes = [112,104,102]
        X_set = []
        for i in range(nb_samples):
                thresholds = np.zeros(2) # all threshold
                probs = np.zeros(2) # all probabilities

                for j in range(2):
                        thresholds[j] = random.random()
                        probs[j] = random.random()

                random.shuffle(classes)
                print(classes)
                print(thresholds)

                X = []
                for k in range(x_size):
                        for j in range(2):
                                probs[j] = random.random()
                        for j in range(2):
                                if j < 1:
                                        if probs[j] > thresholds[j]:
                                                X.append(classes[j])
                                                break
                                else:
                                        if probs[j] > thresholds[j]:
                                                X.append(classes[j])
                                        else:
                                                X.append(classes[j+1])

                X_set.append(copy.deepcopy(X))

        return X_set


# Second version of the saving function
# collect the simulation results to produce the final dataset
# . X_list specifies the list of X to save (corresponding to simulated NoC)
# . start is the first X index in the original X_set
# . NoList is the list of index of X t ignore (e.g. because of simulation error) 

def save_dataset_inter(A, start, X_list, folder_path, data_file, NoList):
        nbR = len(A)
        dataset = []
        filename = data_file
        for idx in range(len(X_list)):
                i = idx + start
                #print(i)
                if i in NoList:
                        print("Nope:", i)
                        continue
                else:
                        pass
                print(i)
                X_mat = X_list[idx]
                ## latency:
                lat_file = str(folder_path) + "sim_"+str(i)+"/latencies/__end2endLatency.stat"
                in_file = open(lat_file, 'r')
                X=[]
                Y=[]
                while(1):
                        #parse input file
                        line = in_file.readline()
                        if (len(line) == 0):
                                break
                        fields = line.split(";")
                        X.append(float(fields[0]))
                        Y.append(float(fields[1]))
                        if(float(fields[0]) >= 0.7):
                                break
                latency = [X,Y]
                #print(latency)

        ## power
        ##- global
                print("\n POWER \n")
                pow_file = str(folder_path) + "sim_"+str(i)+"/powers/__globalPower.stat"
                in_file = open(pow_file, 'r')
                X1=[]
                Y1=[]
                while(1):
                        #parse input file
                        line = in_file.readline()
                        if (len(line) == 0):
                                break
                        fields = line.split(";")
                        X1.append(float(fields[0]))
                        Y1.append(float(fields[1]))
                        if(float(fields[0]) >= 0.7):
                                break
                total_power = [X1,Y1]
                #print(total_power)

                ###- individual
                powers = []
                for j in range(len(A)):
                        pow_file = str(folder_path) + "sim_"+str(i)+"/powers/__router_"+str(j)+"_Power.stat"
                        in_file = open(pow_file, 'r')
                        X2=[]
                        Y2=[]
                        while(1):
                                #parse input file
                                line = in_file.readline()
                                if (len(line) == 0):
                                        break
                                fields = line.split(";")
                                X2.append(float(fields[0]))
                                Y2.append(float(fields[1]))
                                if(float(fields[0]) >= 0.7):
                                        break
                        r_power = [X2,Y2]
                        powers.append(copy.deepcopy(r_power))
                        #print(r_power)
                #print(powers)

                ### area
                print("\n AERA \n")
                area_file = str(folder_path) + "sim_"+str(i)+"/area/area.area"
                #print(area_file)
                in_file = open(area_file, 'r')
                areas=[]
                cpt_line = 0
                while(1):
                        #parse input file
                        line = in_file.readline()
                        #print(line)
                        if cpt_line == 0:
                                line = in_file.readline()
                                #print(line)

                        if (len(line) == 0):
                                break
                        fields = line.split(";")
                        areas.append(float(fields[4]))
                        if(float(fields[0]) >= (nbR-1)):
                                break
                        cpt_line += 1
                total_area = sum(areas)
                #print(areas)
                #print(total_area)

                ### jouls/bytes
                print("\n JOULSpFILTS \n")
                jouls_file = str(folder_path) + "sim_"+str(i)+"/jouls/__efficiency.stat"
                in_file = open(jouls_file, 'r')
                Xj=[]
                Yj=[]
                while(1):
                        #parse input file
                        line = in_file.readline()
                        if (len(line) == 0):
                                break
                        fields = line.split(";")
                        Xj.append(float(fields[0]))
                        Yj.append(float(fields[1]))
                        if(float(fields[0]) >= 0.7):
                                break
                total_jouls = [Xj,Yj]
                #print(total_jouls)

                ### add to dataset
                data = Data_AX(A, X_mat, latency, total_power, powers, total_area, areas, total_jouls)
                #print(data)
                dataset.append(data)
                ### Clean repertory


        with open(filename, 'wb') as f:
                pickle.dump(dataset, f)

        print(dataset)


# saving function, calls save_dataset_inter(), after computing the NoList variable.
def save_inter(start, stop, dataset_folder, X_set):
    NoList = []
    for i in range(start, stop,1):

        filename = dataset_folder+"sim_"+str(i)+"/latencies/__end2endLatency.stat"
        #print(filename)
        try:
            f = open(filename)
        except IOError:
            print("Latency file not accessible")
            NoList.append(i)

        finally:
            f.close()

        filename = dataset_folder+"sim_"+str(i)+"/powers/__router_0_Power.stat"
        #print(filename)
        try:
            f = open(filename)
        except IOError:
            print("Power file not accessible")
            NoList.append(i)

        finally:
            f.close()
    #print("nolist = ", NoList)

    data_file = dataset_folder+ "dataset_inter_"+str(start)+"to"+str(stop)

    A = A_mesh3x4
    save_dataset_inter(A, start, X_set[start:stop], dataset_folder, data_file, NoList)
    #print(len(A))


    ### Save NoList to later complete the dataset
    #print(NoList)
    no_path = dataset_folder+ "noList_inter_"+str(start)+"to"+str(stop)
    with open(no_path, 'wb') as f:
        pickle.dump(NoList, f)


    ### Delete original folders of simulation results to free memory space.
    for i in range(start, stop,1):
        command = str("rm -r "+dataset_folder+"sim_"+str(i))
        print(command)
        os.system(command)

# Launch the simulation of all X within X_set
# simulations are dynamically distributed among a set of threads. The number of threads is determined by n_thread 
def generate_dataset_fromX_dynamic(X_set, dataset_folder):
	A = A_mesh3x4
	n_thread = 40## number of thread to parallelize runs
	cpt = 0
	running_threads_table = [-1]*n_thread
	start = 0
	for i in range(0, len(X_set), 1):
		print("############### \n i = ",i,"\n #################")
		### intermediate dataset saving after each 1000 simulations.
		if i%1000 == 0:
			print("start saving at: ",start)
			save_inter(start,i,dataset_folder, X_set)
			start = i
		#print("i = ", i, " __ X = ", X_set[i])
		n_thread -= 1
		simuID = -1
		while(simuID == -1):
			for thid in range(0,len(running_threads_table)):
				if running_threads_table[thid] == -1:
					simuID = thid
					break
				else:
					if running_threads_table[thid].is_alive() == False:
						running_threads_table[thid] = -1
						simuID = thid
						break

		x = Process(target=thread_function, args=(simuID+1,X_set[i], A,i,dataset_folder,))
		x.start()
		running_threads_table[simuID] = x
	
	#wait for all threads complete
	for thid in range(0, len(running_threads_table)):
		if running_threads_table[thid] != -1 and running_threads_table[thid].is_alive():
			running_threads_table[thid].join()

	save_inter(start,i,dataset_folder, X_set)


def thread_function(simuID, dataX, dataA, n,dataset_folder):

	ned_writer.write_simuID_3x4(dataX, simuID)
	nedpath = hnocs_path+'examples:'+hnocs_path+'src'
	library = hnocs_path+'src/hnocs'
	inifile = hnocs_path+'simulations/simu'+str(simuID)+'/omnetpp.ini'
	
	command = str("opp_run -m -u Cmdenv -c General -n "+nedpath+" -l "+library+" "+inifile)
	print(command)

	os.system(command)

	output_dir = str(dataset_folder) + "sim_"+str(n)
	extract_results(simuID, output_dir)



def generate_dataset_fromX(X_set, dataset_folder):
	### constante Adjacency matrix A
	A = A_mesh3x4

	n_thread = 40## number of thread to parallelize runs

	cpt = 0
	for i in range(0,len(X_set), n_thread):
		
		processes = []
		for j in range(n_thread):
			idx = i+j
			
			if idx >= len(X_set):
				break
			
			simuID = j+1
			X = X_set[idx]
			ned_writer.write_simuID_3x4(X, simuID)
			nedpath = hnocs_path+'examples:'+hnocs_path+'src'
			library = hnocs_path+'src/hnocs'
			inifile = hnocs_path+'simulations/simu'+str(simuID)+'/omnetpp.ini'
			
			p = subprocess.Popen(['opp_run', '-m', '-u', 'Cmdenv', '-c', 'General', '-n', nedpath, '-l', library, inifile])
			processes.append(p)
		# wait for all processes	
		for process in processes:
	  		process.wait()
		print("END SIMU")
		
		### save results
		proc = []
		for j in range(n_thread):
			idx = i+j
			if idx >= len(X_set):
				break
			simuID = j+1
			folder = str(dataset_folder) + "sim_"+str(idx) 
			
			p = Process(target=extract_results, args=(simuID, folder))
			
			proc.append(p)
			
			p.start()

			cpt +=1
		for p in proc:
			p.join()

		#if cpt%(n_thread*5) == 0: # intermediate saving, every 5*n_thread runs
		#	folder_path = str(dataset_folder)
		#	save_dataset(A, X_set[0:cpt], folder_path)

	folder_path = str(dataset_folder)
	save_dataset(A, X_set, folder_path)
	
def extract_results(simuID, folder):
	command = "mkdir " + str(folder)
	print(command)
	command += " && mv "+ hnocs_path +"simulations/simu"+str(simuID)+"/results/* " + str(folder) + "/."
	print(command)
	os.system(command)
	command = "mkdir "+ str(folder)+"/area" + " && mv "+ str(folder)+"/area.area " + str(folder)+"/area/."
	os.system(command)

	# create .stat files
	command = "python latency_FIR_extract.py -i " + str(folder) +"/ -o " + str(folder)+"/latencies/"
	os.system(command)
	command = "python power_FIR_extract.py -i " + str(folder) +"/ -o " + str(folder) + "/powers/"
	os.system(command)
	command = "mkdir "+str(folder) + "/jouls/"
	os.system(command)
	command = "python jouls_FIR_compute.py -s" + str(folder) + "/ -p "+ str(folder) + "/powers/__globalPower.stat"+ " -o " + str(folder) + "/jouls/__efficiency.stat"
	os.system(command)

	# saving lmemory space --> delete useles files after the data extraction (i.e. power history and 2 repetitions (out of 3)
	command = "rm "+ str(folder)+"/power.*"
	os.system(command)
	command = "rm "+ str(folder)+"/General*1.sca"
	os.system(command)
	command = "rm "+ str(folder)+"/General*2.sca"
	os.system(command)



### 





def main():
	

	nb_samples = 10000
	c=4
	r=3
	
	### Define X:
	X_set = generate_random_Xset_3classes(nb_samples, c, r)
	
	
	
	### define folder path so save dataset:
	dataset_folder = "Data/collect_folder/"
	

	## Save the set of X matrices
	X_path = dataset_folder + "X_set"
	with open(X_path, 'wb') as f:
		pickle.dump(X_set, f)

	
	### Generate the dataset
	generate_dataset_fromX_dynamic(X_set, dataset_folder) # create dataset from X_set




main()











