import sys, getopt
import numpy as np
from collections import namedtuple
from collections import OrderedDict
import os
import subprocess
import matplotlib
import matplotlib.pyplot as plt
import glob

clock_ns = 2.0 #ns


def main(argv):
    inputDir = ''
    outputDir = ''
    try:
        opts, args = getopt.getopt(argv,"hi:o:",["ifile="])
    except getopt.GetoptError:
        print("latency_FIR_plot.py -i <input directory> -o <output directory>")
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print("latency_FIR_plot.py -i <input directory> -o <output directory>")
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputDir = arg
        elif opt in ("-o", "--ofile"):
            outputDir = arg
    print("Input Dir is ", inputDir)
    print("Output Dir is ", outputDir)

    parent_dir = os.getcwd()
    path = os.path.join(parent_dir, outputDir)
    
    print("Path : ", path)

    if not os.path.exists(path):
        os.makedirs(path)
    outputfilePath = str(path+"__networkLatency.stat") #str(path+"/"+outputDir+"__networkLatency.stat")
    networkLatencyFile = open(outputfilePath, "w")
    outputfilePath2 = str(path+"__end2endLatency.stat") #str(path+"/"+outputDir+"__end2endLatency.stat")
    end2endLatencyFile = open(outputfilePath2, "w")

    filePattern = inputDir+"General-FID=*.sca"
    scalarFilesList = glob.glob(filePattern)
   
    print(scalarFilesList)

    network_latency_dict = OrderedDict()
    nb_repeat_network_latency_dict = {}
    end_to_end_latency_dict = OrderedDict()
    nb_repeat_end_to_end_latency_dict = {}
    
    for fileP in scalarFilesList:
        in_file = open(fileP, 'r')
        fields = fileP.split("-")
        ufields = fields[1].split("=")
        FID = float(ufields[1])
        FIR = clock_ns/FID
        nb_mean_network_latency = 0
        sum_mean_network_latency = 0.0
        network_latency_ns = False
        nb_mean_end_to_end_latency_ns = 0
        sum_mean_end_to_end_latency_ns = 0.0
        end_to_end_latency_ns = False
        while(1):
            #parse input file
            line = in_file.readline()
           
            if (len(line) == 0):
                break

            if (" network-latency-ns" in line):
                network_latency_ns = True
                end_to_end_latency_ns = False
            elif (" end-to-end-latency-ns" in line):
                network_latency_ns = False
                end_to_end_latency_ns = True
            elif ("statistic" in line) :
                network_latency_ns = False
                end_to_end_latency_ns = False
            elif network_latency_ns and ("mean" in line):
                fields = line.split(" ");
                if "nan" in fields[2]:
                    continue
                current_mean_network_latency = float(fields[2])/clock_ns
                sum_mean_network_latency += current_mean_network_latency
                nb_mean_network_latency +=1
            elif end_to_end_latency_ns and ("mean" in line):
                fields = line.split(" ");
                if "nan" in fields[2]:
                    continue
                current_mean_end_to_end_latency_ns = float(fields[2])/clock_ns
                sum_mean_end_to_end_latency_ns += current_mean_end_to_end_latency_ns
                nb_mean_end_to_end_latency_ns +=1
                

        #print("FID = ", FID, "ns")
        #print("FIR = ", FIR*100, "%")
        globlal_mean_network_latency = sum_mean_network_latency/nb_mean_network_latency
        #print("globlal_mean_network_latency = ", globlal_mean_network_latency)
        global_end_to_end_latency_ns = sum_mean_end_to_end_latency_ns/nb_mean_end_to_end_latency_ns
        #print("global_end_to_end_latency_ns = ", global_end_to_end_latency_ns)
        if FIR in nb_repeat_network_latency_dict.keys():
            nb_repeat_network_latency_dict[FIR] +=1
            network_latency_dict[FIR] += globlal_mean_network_latency
        else:
            nb_repeat_network_latency_dict[FIR] = 1
            network_latency_dict[FIR] = globlal_mean_network_latency
            
        if FIR in nb_repeat_end_to_end_latency_dict.keys():
            nb_repeat_end_to_end_latency_dict[FIR] += 1
            end_to_end_latency_dict[FIR] += global_end_to_end_latency_ns
        else:
            nb_repeat_end_to_end_latency_dict[FIR] = 1
            end_to_end_latency_dict[FIR] = global_end_to_end_latency_ns


    for FIR in nb_repeat_network_latency_dict.keys():          
        network_latency_dict[FIR] = network_latency_dict[FIR]/nb_repeat_network_latency_dict[FIR]

    for FIR in nb_repeat_end_to_end_latency_dict.keys():
        end_to_end_latency_dict[FIR] = end_to_end_latency_dict[FIR]/nb_repeat_end_to_end_latency_dict[FIR]
            
    ord_network_latency_list = sorted((float(x),y) for x,y in network_latency_dict.items())
    print("network_latency")
    for i in range(0, len(ord_network_latency_list)):
        print("FIR=",ord_network_latency_list[i][0], "net lat = ", ord_network_latency_list[i][1])
        networkLatencyFile.write(str(ord_network_latency_list[i][0])+";"+str(ord_network_latency_list[i][1])+"\n")

    ord_end_to_end_latency_list = sorted((float(x),y) for x,y in end_to_end_latency_dict.items())
    print("end_to_end_latency")
    for i in range(0, len(ord_end_to_end_latency_list)):
        print("FIR=",ord_end_to_end_latency_list[i][0], "net lat = ", ord_end_to_end_latency_list[i][1])
        end2endLatencyFile.write(str(ord_end_to_end_latency_list[i][0])+";"+str(ord_end_to_end_latency_list[i][1])+"\n")
    
    return 0


if __name__ == "__main__":
    main(sys.argv[1:])
