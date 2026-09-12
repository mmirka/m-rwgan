import sys, getopt
import numpy as np
from collections import namedtuple
from collections import OrderedDict
import os
import subprocess
import matplotlib
import matplotlib.pyplot as plt
import glob

clock_ns = 2.0


def main(argv):
    inputDir = ''
    outputDir = ''
    try:
        opts, args = getopt.getopt(argv,"hi:o:",["ifile="])
    except getopt.GetoptError:
        print("power_FIR_plot.py -i <input directory> -o <output directory>")
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print("power_FIR_plot.py -i <input directory> -o <output directory>")
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
    outputfilePath = str(path+"__globalPower.stat") #str(path+"/"+outputDir+"__globalPower.stat")
    globalPowerFile = open(outputfilePath, "w")

    filePattern = inputDir+"/power.*"
    scalarFilesList = glob.glob(filePattern)
   
    print(scalarFilesList)

    fir_power_dict = {}
    nb_repeat_power_dict = {}
    for fileP in scalarFilesList:
        print(fileP)
        in_file = open(fileP, 'r')
        fields = fileP.split(".")
        FID = float(str(fields[1]+"."+fields[2]))
        FIR = clock_ns/FID
        
        power_routers_dict = {}
        router_id = 0
        while(1):
            #parse input file
            line = in_file.readline()
           
            if (len(line) == 0):
                break
            if ("#Router" in line): #header
                continue

            fields = line.split(";")

            if (len(fields) < 4) :
                print("warning len line = ", line)
                continue

            if (len(fields[0]) != 0):
                router_id = int(fields[0])
                
            power = float(fields[-1])

            if router_id in power_routers_dict.keys():
                power_routers_dict[router_id] += power
            else:
                power_routers_dict[router_id] = power
        

        #print("fir_power_dict = ",fir_power_dict)
        if FIR in fir_power_dict.keys():
            nb_repeat_power_dict[FIR] += 1
            for router in power_routers_dict.keys():
                fir_power_dict[FIR][router] += power_routers_dict[router]
        else:
            nb_repeat_power_dict[FIR] = 1
            fir_power_dict[FIR] = power_routers_dict
                


    ite = 0
    for FIR in sorted(fir_power_dict.keys()):
        gPower = 0.0
        for router in fir_power_dict[FIR].keys():
            outputRouterFilePath = str(path+"__router_"+str(router)+"_Power.stat") #str(path+"/"+outputDir+"__router_"+str(router)+"_Power.stat")
            if (ite == 0):
                outputRouterFile = open(outputRouterFilePath, "w")
            else:
                outputRouterFile = open(outputRouterFilePath, "a")
            outputRouterFile.write(str(FIR)+";"+ str(fir_power_dict[FIR][router]/nb_repeat_power_dict[FIR])+"\n")
            print("FIR = ", FIR, "power = ", fir_power_dict[FIR][router]/nb_repeat_power_dict[FIR])
            outputRouterFile.flush()
            gPower += fir_power_dict[FIR][router]/nb_repeat_power_dict[FIR]
        ite += 1

        globalPowerFile.write(str(FIR)+";"+ str(gPower)+"\n")
               
    return 0


if __name__ == "__main__":
    main(sys.argv[1:])
