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
#import routers

#from routers import full_DIC_R as DIC_R

import copy

import subprocess

from multiprocessing import Process, Pool

import time

#################### to adapt #########################
#hnocs_path = "<HNOCS_ROOT>/" 
#######################################################

### 4x4 X by hand:
#     0   1   2   3
#
# 0   o - o - o - o
#     -   -   -   -
# 1   o - o - o - o
#     -   -   -   -
# 2   o - o - o - o
#     -   -   -   -
# 3   o - o - o - o

A_mesh4x4 = [[0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
             [1,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0],
             [0,1,0,1,0,0,1,0,0,0,0,0,0,0,0,0],
             [0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0],
             [1,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0],
             [0,1,0,0,1,0,1,0,0,1,0,0,0,0,0,0],
             [0,0,1,0,0,1,0,1,0,0,1,0,0,0,0,0],
             [0,0,0,1,0,0,1,0,0,0,0,1,0,0,0,0],
             [0,0,0,0,1,0,0,0,0,1,0,0,1,0,0,0],
             [0,0,0,0,0,1,0,0,1,0,1,0,0,1,0,0],
             [0,0,0,0,0,0,1,0,0,1,0,1,0,0,1,0],
             [0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,1],
             [0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0],
             [0,0,0,0,0,0,0,0,0,1,0,0,1,0,1,0],
             [0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,1],
             [0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0]]
from A_mesh8x8 import A_mesh8x8

print(A_mesh8x8)


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


#######################################################################################################################################
########                       Examples of X liste:
#######################################################################################################################################
X_HP = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

X_LP = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

X_MP = [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8]

X_ULP = [9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9]

X_ILP = [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10]

X_AngHP = [3, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 4]

X_AngLP = [3, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 5, 1, 1, 4]

X_AngSideHP = [3, 7, 7, 2, 6, 0, 0, 6, 6, 0, 0, 6, 5, 7, 7, 4]

X_AngSideLP = [3, 7, 7, 2, 6, 1, 1, 6, 6, 1, 1, 6, 5, 7, 7, 4]


X_AngHP2 = [13, 0, 0, 12, 0, 0, 0, 0, 0, 0, 0, 0, 15, 0, 0, 14]

X_AngLP2 = [13, 1, 1, 12, 1, 1, 1, 1, 1, 1, 1, 1, 15, 1, 1, 14]

X_AngSideHP2 = [13, 17, 17, 12, 16, 0, 0, 16, 16, 0, 0, 16, 15, 17, 17, 14]

X_AngSideLP2 = [13, 17, 17, 12, 16, 1, 1, 16, 16, 1, 1, 16, 15, 17, 17, 14]


X_transpose_1 = [1, 7, 7, 1, 6, 3, 2, 6, 6, 5, 4, 6, 1, 7, 7, 1]
X_transpose_2 = [1, 17, 17, 1, 16, 13, 12, 16, 16, 15, 14, 16, 1, 17, 17, 1]
X_transpose_3 = [1, 27, 27, 1, 26, 23, 22, 26, 26, 25, 24, 26, 1, 27, 27, 1]
X_transpose_4 = [1, 37, 37, 1, 36, 33, 32, 36, 36, 35, 34, 36, 1, 37, 37, 1]

X_hotspot0 = [112, 17, 17, 17, 112, 17, 17, 17, 112, 17, 17, 17, 112, 17, 17, 17]


X_HP_8 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

X_LP_8 = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

X_AngHP_8 = [3, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 4]

X_AngLP_8 = [3, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 5, 1, 1, 1, 1, 1, 1, 4]

X_AngSideHP_8 = [3, 7, 7, 7, 7, 7, 7, 2, 6, 0, 0, 0, 0, 0, 0, 6, 6, 0, 0, 0, 0, 0, 0, 6, 6, 0, 0, 0, 0, 0, 0, 6, 6, 0, 0, 0, 0, 0, 0, 6, 6, 0, 0, 0, 0, 0, 0, 6, 6, 0, 0, 0, 0, 0, 0, 6, 5, 7, 7, 7, 7, 7, 7, 4]

X_AngSideLP_8 = [3, 7, 7, 7, 7, 7, 7, 2, 6, 1, 1, 1, 1, 1, 1, 6, 6, 1, 1, 1, 1, 1, 1, 6, 6, 1, 1, 1, 1, 1, 1, 6, 6, 1, 1, 1, 1, 1, 1, 6, 6, 1, 1, 1, 1, 1, 1, 6, 6, 1, 1, 1, 1, 1, 1, 6, 5, 7, 7, 7, 7, 7, 7, 4]

X8x8_HOMO_2 = [102] * 64
X8x8_HOMO_4 = [104] * 64
X8x8_HOMO_12 = [112] * 64
X8x8_HOMO_8 = [108] * 64
X8x8_HOMO_16 = [116] * 64


X_HOMO_1 = [101] * 16
X_HOMO_2 = [102] * 16
X_HOMO_3 = [103] * 16
X_HOMO_4 = [104] * 16
X_HOMO_5 = [105] * 16
X_HOMO_6 = [106] * 16
X_HOMO_7 = [107] * 16
X_HOMO_8 = [108] * 16
X_HOMO_9 = [109] * 16
X_HOMO_10 = [110] * 16
X_HOMO_11 = [111] * 16
X_HOMO_12 = [112] * 16
X_HOMO_13 = [113] * 16
X_HOMO_14 = [114] * 16
X_HOMO_15 = [115] * 16
X_HOMO_16 = [116] * 16
X_HOMO_17 = [117] * 16
X_HOMO_18 = [118] * 16
X_HOMO_19 = [119] * 16
X_HOMO_20 = [120] * 16

liste_HOMO = [X_HOMO_1, X_HOMO_2, X_HOMO_3, X_HOMO_4, X_HOMO_5, X_HOMO_6, X_HOMO_7, X_HOMO_8, X_HOMO_9, X_HOMO_10, X_HOMO_11, X_HOMO_12, X_HOMO_13, X_HOMO_14, X_HOMO_15, X_HOMO_16, X_HOMO_17, X_HOMO_18, X_HOMO_19, X_HOMO_20]
liste_HOMO_1_12 = [X_HOMO_1, X_HOMO_2, X_HOMO_3, X_HOMO_4, X_HOMO_5, X_HOMO_6, X_HOMO_7, X_HOMO_8, X_HOMO_9, X_HOMO_10, X_HOMO_11, X_HOMO_12]
X_local_1 = [201] * 16
X_local_2 = [202] * 16
X_local_3 = [203] * 16
X_local_4 = [204] * 16
X_local_5 = [205] * 16
X_local_6 = [206] * 16
X_local_7 = [207] * 16
X_local_8 = [208] * 16
X_local_9 = [209] * 16
X_local_10 = [210] * 16
X_local_11 = [211] * 16
X_local_12 = [212] * 16
X_local_13 = [213] * 16
X_local_14 = [214] * 16
X_local_15 = [215] * 16
X_local_16 = [216] * 16
X_local_17 = [217] * 16
X_local_18 = [218] * 16
X_local_19 = [219] * 16
X_local_20 = [220] * 16

liste_local = [X_local_1, X_local_2, X_local_3, X_local_4, X_local_5, X_local_6, X_local_7, X_local_8, X_local_9, X_local_10, X_local_11, X_local_12, X_local_13, X_local_14, X_local_15, X_local_16, X_local_17, X_local_18, X_local_19, X_local_20]
liste_NSEW_12_L_2to12 = [X_local_2, X_local_3, X_local_4, X_local_5, X_local_6, X_local_7, X_local_8, X_local_9, X_local_10, X_local_11, X_local_12]
X_nsew_1 = [301] * 16
X_nsew_2 = [302] * 16
X_nsew_3 = [303] * 16
X_nsew_4 = [304] * 16
X_nsew_5 = [305] * 16
X_nsew_6 = [306] * 16
X_nsew_7 = [307] * 16
X_nsew_8 = [308] * 16
X_nsew_9 = [309] * 16
X_nsew_10 = [310] * 16
X_nsew_11 = [311] * 16
X_nsew_12 = [312] * 16
X_nsew_13 = [313] * 16
X_nsew_14 = [314] * 16
X_nsew_15 = [315] * 16
X_nsew_16 = [316] * 16
X_nsew_17 = [317] * 16
X_nsew_18 = [318] * 16
X_nsew_19 = [319] * 16
X_nsew_20 = [320] * 16

liste_nsew = [X_nsew_1, X_nsew_2, X_nsew_3, X_nsew_4, X_nsew_5, X_nsew_6, X_nsew_7, X_nsew_8, X_nsew_9, X_nsew_10, X_nsew_11, X_nsew_12, X_nsew_13, X_nsew_14, X_nsew_15, X_nsew_16, X_nsew_17, X_nsew_18, X_nsew_19, X_nsew_20]

liste_HandX = [X_HP, X_LP, X_AngHP, X_AngLP, X_AngSideHP, X_AngSideLP]
liste_HandX2 = [X_AngHP2, X_AngLP2, X_AngSideHP2, X_AngSideLP2]
liste_fullR_X = [X_HP, X_ULP, X_LP]

liste_test2 = [X_HP_8, X_LP_8, X_AngHP_8, X_AngLP_8, X_AngSideHP_8, X_AngSideLP_8]
liste_X_transpose = [X_HP, X_LP, X_ILP, X_transpose_1, X_transpose_2]
liste_X_hotspot = [X_hotspot0, X_HOMO_12, X_HOMO_4, X_HOMO_8]

liste8x8_HOMO_3c = [X8x8_HOMO_12, X8x8_HOMO_4, X8x8_HOMO_2]#, X8x8_HOMO_12, X8x8_HOMO_4, X8x8_HOMO_2]#, X8x8_HOMO_16, X8x8_HOMO_8]

#### 8c_DIC topo _ 8c_DIC = {0 : "HP", 1 : "LP", 2 : "ANE2", 3 : "ANW2", 4 : "ASE2", 5 : "ASW2", 6 : "DNS2", 7 : "DEW2"}
HP = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
LP = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
uni_HP_angles = [5, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 2]
uni_LP_angles = [5, 1, 1, 4, 1, 1, 1, 1, 1, 1, 1, 1, 3, 1, 1, 2]
uni_HP_angles_opp = [3, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 4]
uni_LP_angles_opp = [3, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 5, 1, 1, 4]
trans1 = [1, 7, 7, 1, 6, 5, 4, 6, 6, 3, 2, 6, 1, 7, 7, 1]
trans1_opp = [1, 7, 7, 1, 6, 3, 2, 6, 6, 5, 4, 6, 1, 7, 7, 1]
trans_HP_sides = [0, 7, 7, 0, 6, 0, 0, 6, 6, 0, 0, 6, 0, 7, 7, 0]
trans_LP_sides = [1, 7, 7, 1, 6, 1, 1, 6, 6, 1, 1, 6, 1, 7, 7, 1]


liste_uni_angles = [HP, LP, uni_HP_angles, uni_LP_angles]
liste_uni_angles_opp = [HP, LP, uni_HP_angles_opp, uni_LP_angles_opp]
liste_trans = [HP, LP, trans1, trans_HP_sides, trans_LP_sides]
liste_trans_opp = [HP, LP, trans1_opp, trans_HP_sides, trans_LP_sides]

#######################################################################################################################################
########                       
#######################################################################################################################################


def create_random_X(size):

    limit = len(DIC_R) - 1

    X = np.zeros(size)

    for i in range(size):
    	X[i] = randint(0, limit)

    return X


def save_dataset(A, X_list, folder_path, data_file, NoList):
	nbR = len(A)
	dataset = []
	filename = data_file
	for i in range(len(X_list)):
                
		#print(i)
		if i in NoList:
			print("Nope:", i)
			continue
		else:
			pass
		print(i)
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
	
	
def read_dataset(folder_path):
	filename = folder_path+"dataset"
	with open(filename, 'rb') as f:
		dataset = pickle.load(f)

	for data in dataset:
		print(data.A)
		print(data.X)
		print(data.latency )
		print(data.total_power )
		print(data.powers )
		print(data.total_area)
		print(data.areas)




def check_isIn(sample, dataset):
	isIn = False
	for elem in dataset:
		if len(elem) != len(sample):
			print(" Warning: Not same size")
			pass
		else:
			isSame = True
			for i in range(len(sample)):
				if sample[i] != elem[i]:
					isSame = False
					break
			if isSame:
				isIn = True
				break

	return isIn


def generate_random_Xset(nb_samples, c, r):
	x_size = c*r
	X_set = []
	for i in range(nb_samples):
		in_set = True
		while(in_set):
			X = create_random_X(x_size)
			in_set = check_isIn(X, X_set)
		X_set.append(copy.deepcopy(X))

	return X_set

def generate_random_Xset_8classes(nb_samples, c, r):
	x_size = c*r
	classes = [104,12,13,14,15,16,17,112] 
	X_set = []
	for i in range(nb_samples):
		thresholds = np.zeros(7) # all threshold
		probs = np.zeros(7) # all probabilities

		for j in range(7):
			thresholds[j] = random.random()
			probs[j] = random.random()

		random.shuffle(classes)
		print(classes)
		print(thresholds)

		X = []
		for k in range(x_size):
			for j in range(7):
				probs[j] = random.random()
			for j in range(7):
				if j < 6:
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


def generate_dataset_fromX(X_set, dataset_folder):
	### constante Adjacency matrix A
	A = A_mesh4x4

	n_thread = 20 ## number of thread to parallelize runs

	cpt = 0
	for i in range(0,len(X_set), n_thread):
		
		processes = []
		for j in range(n_thread):
			idx = i+j
			
			if idx >= len(X_set):
				break
			
			simuID = j+1
			X = X_set[idx]
			ned_writer.write_simuID(X, simuID)
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

		if cpt%(n_thread*5) == 0: # intermediate saving, every 5*n_thread runs
			folder_path = str(dataset_folder)
			save_dataset(A, X_set[0:cpt], folder_path)

	folder_path = str(dataset_folder)
	save_dataset(A, X_set, folder_path)
	
def extract_results(simuID, folder):
	command = "mkdir " + str(folder)
	#print(command)
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

	command = "rm "+ str(folder)+"/*"
	os.system(command)
	
	#time.sleep(3)
	print("ENDENDEND")

### 
def test_fromX(X_set, dataset_folder):
        print(X_set)
        X_test = X_set[0]
        l_X = len(X_set)
        l = len(X_test)
        nb = int(math.sqrt(l))
        if nb == 4:
                A = A_mesh4x4
        elif nb == 8:
                A = A_mesh8x8
        else:
                print("ERROR len(X) = ", l)
                A = 0

        cpt = 0
        for i in range(l_X):

                processes = []
                X = X_set[i]
                simuID = 4
                print(X)

                ned_writer.write_simuID(X, simuID)
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
                
                folder = str(dataset_folder) + "sim_"+str(i)
                command = "mkdir " + str(folder)
                #print(command)
                command += " && mv "+hnocs_path+"simulations/simu"+str(simuID)+"/results/* " + str(folder) + "/."
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

                #command = "rm "+ str(folder)+"/*"
                #os.system(command)

                               
        folder_path = str(dataset_folder)
        save_dataset(A, X_set, folder_path)




def main():
    print("Start")
    nb_samples = 10000
    c=8
    r=8
	
    ### Define X:
    #X_set = liste8x8_HOMO_3c
    #X_set = [X8x8_HOMO_2]
	
    ## e.g.
    #X_set = generate_random_Xset_nbClass(nb_samples, c, r)
    #X_set = generate_random_Xset_nbClass(nb_samples, c, r)  # create random X_set
    #X_set = liste_NSEW_12_L_2to12
	
    ### define folder path so save dataset:
    dataset_folder = "Data/collect_all_12r3c_uniform/"
    X_path = "allX_12r_3c"
    print("load X_setall")
    with open(X_path, 'rb') as f:
        X_setall = pickle.load(f)
    print("X_setall loaded")
    X_set = X_setall[0:17000]
    print(len(X_set))
    #nb_data = 11430
    ##### IF JOULS Deleted
    #for i in range(nb_data):
    #        print(i)
    #        folder = str(dataset_folder) + "sim_"+str(i)
    #        command = "mkdir "+str(folder) + "/jouls/"
    #        os.system(command)
    #        command = "python jouls_FIR_compute.py -s" + str(folder) + "/ -p "+ str(folder) + "/powers/__globalPower.stat"+ " -o " + str(folder) + "/jouls/__efficiency.stat"
    #        os.system(command)
    ######

    NoList = []
    for i in range(len(X_set)):
        if i%100==0:
            print(i)
        #filename = dataset_folder+"sim_"+str(i)+"/General-FID=50-#0.sca"
        #print(filename)
        #try:
        #    f = open(filename)
        #except IOError:
        #    print("General file not accessible")
        #    NoList.append(i)
        #    
        #finally:
        #    f.close()

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
    print("nolist = ", NoList)



    data_file = "Data/collect_all_12r3c_uniform/dataset_0to17k"

    A = A_mesh3x4
    save_dataset(A, X_set, dataset_folder, data_file, NoList)
    print(len(A))

    print(NoList)
    no_path = "Data/collect_all_12r3c_uniform/noList_0to17k"
    with open(no_path, 'wb') as f:
        pickle.dump(NoList, f)

	
   ## e.g.
    #dataset_folder = "<DATA_ROOT>/test_NSEW_12_L_2to12/" # path to save dataset

	
    ## Save the set of X matrices
    #X_path = dataset_folder + "X_set"
    #with open(X_path, 'wb') as f:
    #    pickle.dump(X_set, f)

	
    #test_fromX(X_set, dataset_folder) #for small X_set
    #generate_dataset_fromX(X_set, dataset_folder) # create dataset from X_set




main()


