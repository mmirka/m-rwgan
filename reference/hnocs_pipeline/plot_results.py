import numpy as np
from collections import namedtuple
from collections import OrderedDict
import os
import subprocess
import matplotlib
import matplotlib.pyplot as plt
import glob, pickle


#liste_HandX = [X_HP, X_LP, X_AngHP, X_AngLP, X_AngSideHP, X_AngSideLP]

class Data_AX:
    def __init__(self, A, X, latency, total_power, powers, total_area, areas): 
        self.A = A
        self.X = X
        self.latency = latency
        self.total_power = total_power
        self.powers = powers
        self.total_area = total_area
        self.areas = areas




def main():

	#filename_hand = "<DATA_ROOT>/test_hand_multi/dataset"
	#filename_hand = "<DATA_ROOT>/fullRouters/dataset"
	filename_hand = "Data/test3_12r_uniform/dataset"
	#filename_hand2 = "<DATA_ROOT>/verif_8c_trans_opp/dataset"
	#filename = "<DATA_ROOT>/random_3classes_2/dataset"
	#filename = "<DATA_ROOT>/random_4x4_mesh_dataset/dataset"
	#with open(filename, 'rb') as f:
	#	dataset = pickle.load(f)
	with open(filename_hand, 'rb') as f:
		dataset_hand = pickle.load(f)
	#with open(filename_hand2, 'rb') as f:
	#	dataset_hand2 = pickle.load(f)


	latencies = []
	total_powers = []
	total_areas = []

	latencies_hand = []
	total_powers_hand = []
	total_areas_hand = []
	total_jouls_hand = []

	latencies_hand2 = []
	total_powers_hand2 = []
	total_areas_hand2 = []
	total_jouls_hand2 = []

	# recover data
	#for data in dataset:
	#	latencies.append(data.latency)
	#	total_powers.append(data.total_power)
	#	total_areas.append(data.total_area)

	for data_hand in dataset_hand:
		latencies_hand.append(data_hand.latency)
		total_powers_hand.append(data_hand.total_power)
		total_areas_hand.append(data_hand.total_area)
		total_jouls_hand.append(data_hand.total_jouls)

	#for data_hand2 in dataset_hand2:
	#	latencies_hand2.append(data_hand2.latency)
	#	total_powers_hand2.append(data_hand2.total_power)
	#	total_areas_hand2.append(data_hand2.total_area)
	#	total_jouls_hand2.append(data_hand2.total_jouls)


	# plot latencies
	plt.figure()
	#cpt = 0
	#for lats in latencies:
	#	l = np.asarray(lats)
	#	plt.plot(l[0,:], l[1,:], label = str(cpt), marker='o')
	#	cpt += 1

	cpt = 0
	for lats in latencies_hand:
		l = np.asarray(lats)
		lab = str(cpt+1) 
		
		plt.plot(l[0,:], l[1,:], label = str(lab), marker='x')
		
		cpt += 1

	#cpt = 0
	#for lats in latencies_hand2:
	#	l = np.asarray(lats)
	#	if cpt == 0:
	#		lab = "Trans3"
	#	#plt.plot(l[0,:], l[1,:], label = str(lab), marker='o')
	#	cpt += 1

	plt.ylim(0, 500)
	plt.legend()
	plt.grid()
	


	plt.figure()

	#cpt = 0
	#for pows in total_powers:
	#	l = np.asarray(pows)
	#	#plt.plot(l[0,:], l[1,:], label = str(cpt), marker='o')
	#	cpt += 1

	cpt = 0
	for pows in total_powers_hand:
		l = np.asarray(pows)
		lab = str(cpt+1) 
		
		plt.plot(l[0,:], l[1,:], label = str(lab), marker='x')

		cpt += 1

		

	#cpt = 0
	#for pows in total_powers_hand2:
	#	p = np.asarray(pows)
	#	if cpt == 0:
	#		lab = "Trans3"
	#	#plt.plot(p[0,:], p[1,:], label = str(lab), marker='o')
	#	cpt += 1

	#plt.ylim(0, 500)
	plt.legend()
	plt.grid()


	plt.figure()

	cpt = 0
	for jouls in total_jouls_hand:
		l = np.asarray(jouls)
		lab = str(cpt+1) 
		
		plt.plot(l[0,:], l[1,:], label = str(lab), marker='x')

		cpt += 1


	#cpt = 0
	#for jouls in total_jouls_hand2:
	#	p = np.asarray(jouls)
	#	if cpt == 0:
	#		lab = "Trans3"
	#	#plt.plot(p[0,:], p[1,:], label = str(lab), marker='o')
	#	cpt += 1

	#plt.ylim(0, 500)
	plt.legend()
	plt.grid()

	plt.show()






	return

main()

#filename_hand = "<DATA_ROOT>/test_hand_multi/dataset"
#filename = "<DATA_ROOT>/random_3classes_2/dataset"
#filename = "<DATA_ROOT>/random_4x4_mesh_dataset/dataset"
#with open(filename, 'rb') as f:
#	dataset = pickle.load(f)

#print(len(dataset))
