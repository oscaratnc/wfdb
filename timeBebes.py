import wfdb
import numpy as np 
import os

if __name__ == '__main__': 
    ecg = np.array([])
    ip = np.array([])
    spo2 = np.array([])
    fs =0  
    file  = input("Which record to read? \n")
    while (True):    
       
        fileNum = input("which file you want to read? \n")
        if fileNum == 0:
            break
        if fileNum<=9:
            filenum= "000"+str(fileNum)
        elif fileNum<=99:
            filenum="00"+str(fileNum)
        else:
            filenum="0"+str(fileNum)

        path = "%s\\%s_%s"  %(file,file,filenum)
        print path
        try:
            record  = wfdb.rdsamp(path, channels = [2])
            sampleNumber = len(record.p_signals)
            samplePeriod = 1.0/record.fs
            print "Time Lapse Minutes: ", (sampleNumber*samplePeriod)/60.0
            continue
            

        except:
            print "algo paso mal"
            continue
