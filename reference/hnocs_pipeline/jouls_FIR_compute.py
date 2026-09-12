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
flits_in_packet = 8
bytes_in_flit = 4
simu_time = 0.000050 #s (50us)

def main(argv):
    powerFilePath = ''
    statDirPath = ''
    outputFilePath = ''
    try:
        opts, args = getopt.getopt(argv,"hs:p:o:",["ifile="])
    except getopt.GetoptError:
        print("jouls_FIR_compute.py -s <stat file> -p <power file> -o <outfile file>")
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print("jouls_FIR_compute.py -s <stat file> -p <power file> -o <outfile file>")
            sys.exit()
        elif opt == "-p":
            powerFilePath = arg
        elif opt == "-s":
            statDirPath = arg
        elif opt == "-o":
            outputFilePath = arg
    print("Power file is ", powerFilePath)
    print("Stat file is ", statDirPath)
    print("Output file is ", outputFilePath)

    outputFile = open(outputFilePath, "w")
    
    filePattern = statDirPath+"General-FID=*#0.sca"
    scalarFilesList = glob.glob(filePattern)
   
    print(scalarFilesList)
    sum_sent_pkts_dict = {}
    for fileP in scalarFilesList:
        in_file = open(fileP, 'r')
        fields = fileP.split("-")
        ufields = fields[1].split("=")
        FID = float(ufields[1])
        if(FID == 0):
            continue
        FIR = clock_ns/FID
        print("FIR = ",FIR)
        current_mean = 0
        sum_sent_pkts = 0
        number_sent_packets = False
        #compute all flits sent
        while(1):
            #parse input file
            line = in_file.readline()
            
            if (len(line) == 0):
                break
        
            if (" number-sent-packets" in line):
                number_sent_packets = True
            elif ("statistic" in line) :
                number_sent_packets = False
            elif number_sent_packets and ("mean" in line):
                fields = line.split(" ");
                if "nan" in fields[2]:
                    continue
                current_mean = int(fields[2])
                sum_sent_pkts += current_mean

        sum_sent_pkts_dict[round(FIR, 3)] = sum_sent_pkts


    print(sum_sent_pkts_dict)

    #retrive powers
    FIR_list = []
    jouls_list = []
    powerFile = open(powerFilePath, 'r')
    while(1):
        #parse input file
        line = powerFile.readline()
        
        if (len(line) == 0):
            break

        fields = line.split(";")
        if (len(fields) != 2):
            print("Error line len "+str(len(fields)))
            sys.exit()

        FIR = float(fields[0])
        power = float(fields[1])

        if(FIR == 0):
            continue

        FIR_list.append(FIR)

        round_FIR = round(FIR, 3)
        print("round FIR = ", round_FIR)

        sum_bytes = sum_sent_pkts_dict[round_FIR]*flits_in_packet*bytes_in_flit
        pico_jouls = power*1000000*simu_time
        jouls_list.append(pico_jouls/sum_bytes)
        
        outputFile.write(str(FIR)+";"+str(pico_jouls/sum_bytes)+"\n")

    print(jouls_list)
            
if __name__ == "__main__":
    main(sys.argv[1:])
