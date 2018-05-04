import wfdb
import sys
import numpy as np 
from scipy import signal as sp
import os
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg
import Processing as pro

   
class Records:
        
    if __name__ == '__main__': 

        ecg = np.array([])
        ip = np.array([])
        spo2 = np.array([])
        fs =0      
        file  = input("Which record to read?")
        filenum = input ("What number file?")

        path = "%s\\%s_000%s"  %(file,file,filenum)
        print path

        record  = wfdb.rdsamp(path)
     
        fs = record.fs
        sampleNumber = len(record.p_signals)
        samplePeriod = 1.0/fs
        print "Time Lapse Minutes: ", (sampleNumber*samplePeriod)
            
        ecgp= pro.ECGProcessing()
        gp = pro.generalProcessing()
        
        print "sample Read"
        
        print "SampleFrequency",fs
        Signal1 = record.p_signals
        
        spo2 = record.p_signals[fs*begin:fs*finish,1]
        spo2 = record.p_signals[fs*begin:fs*finish,1]
        print "SPO2",spo2
       
        ecg = record.p_signals[:,2]       
        ecg = gp.removeNAN(ecg)
        begin = input("Begin time:")
        finish = input("Finish time:")
        ecg = ecg[fs*begin:fs*finish]
  
        print len(ecg)
        winECG = fs*5
        x =0
        while x < len(ecg)-winECG:
            ecgp.HRCalculation(ecg[x:x+winECG],fs)
            x= x+winECG
            print x
        print pro.globalV.HR_vector
  
       
            

        