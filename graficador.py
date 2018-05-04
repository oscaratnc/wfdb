
import sys
import numpy as np 

from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg





class Graficador:
    
    def plotSignals(self,initTime, finTime, signal1,signal2,signal3,signal4,signal5,signal6, signal7):
        #Graphing results
        app = QtGui.QApplication([])
        sampleFreqn = 1.
        sampleFreq = 125.
        sampleNum = finTime-initTime
        sampleNumF = ((finTime)-(initTime))*sampleFreq
        print "SAmples: ",sampleNum, sampleNumF
        samplePeriodn = 1./sampleFreqn
        samplePeriodf = 1./sampleFreq
        tn = np.linspace(initTime,finTime,sampleNum-1)
        tfreq = np.linspace(initTime,finTime,sampleNumF-1)
        #print"len tn: ", tn
        #print "len signal1: ",len(signal1)
        #print "len tfreq: ", tfreq
        #print "len signal5: ",len(signal5)
        win = pg.GraphicsWindow() 
        win.resize(500,600)
        win.showMaximized()

        win.setWindowTitle("Neonate Records")
        pg.setConfigOptions(antialias= True)

        p1  = win.addPlot()
        p1.setLabel('bottom',"Samples","mu")
        p1.setLabel('left',"Spo2","%O2")
        #p1.setTitle("Oxygen Saturation")
        
        p1.setYRange(80,120)
        p1.plot(tn,signal1, pen = pg.mkPen(color = 'r', width=1))

        win.nextRow()
        p2  = win.addPlot()
        p2.setLabel('bottom',"Samples","mu")
        p2.setLabel('left',"Heart Rate","bpm")
        #p2.setTitle(" ECG Heart Rate")
        p2.setYRange(100,200)
        p2.plot(tn,signal2, pen = pg.mkPen(color = 'g', width=1))
        
       
        # win.nextRow()
        # p3  = win.addPlot()
        # p3.setLabel('bottom',"Time (t)","s")
        # #p3.setTitle("CARDIAC PULSE")
        # p3.setLabel('left', "Pulse Rate", 'pm')
        # #p3.setYRange(20,90)
        # p3.plot(tn,signal3, pen = pg.mkPen(color = 'w', width=1))
    
        win.nextRow()
        p4 = win.addPlot()
        p4.setLabel('bottom',"Time (t)",'s')
        # # p4.setTitle("Oxygen Saturation")
        p4.setLabel('left', "RR", 'pm')
        p4.setYRange(90,100)
        # # p4.setXRange(0,3000)
        p4. plot(tn,signal4, pen = pg.mkPen(color = 'b', width = 1))

        win.nextRow()
        p5 = win.addPlot()
        p5.plot(tfreq,signal5)
        p5.setYRange(-2,2)
        p5.setLabel('bottom',"Time (t)","s")
        p5.setLabel('left',"ECG","mVNorm")

        win.nextRow()
        p6 = win.addPlot()
        p6.plot(tfreq,signal6)
        p6.setYRange(-1,2)
        p6.setLabel('bottom',"Time (t)","s")
        p6.setLabel('left',"CI","mV Norm")

        win.nextRow()
        p7 = win.addPlot()
        p7.plot(tfreq,signal7)
        p7.setYRange(0,1)
        p7.setLabel('bottom',"Time (t)","s")
        p7.setLabel('left',"PLETH","mV Norm")
        
        app.exec_()


        