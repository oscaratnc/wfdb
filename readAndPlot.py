import wfdb
import sys
import numpy as np  
import os
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg
import Processing as pro


class graficador:
    
    def plotSignals(self,signal1,signal2, signal3):
        #Graphing results
        app = QtGui.QApplication([])

        win = pg.GraphicsWindow() 
        win.resize(500,600)
        #win.showMaximized()

        win.setWindowTitle("Neonate Records")
        pg.setConfigOptions(antialias= True)

        p1  = win.addPlot()
        
        p1.plot(signal1, pen = pg.mkPen(color = 'r', width=1))

        win.nextRow()
        p2  = win.addPlot()
        p2.plot(signal2, pen = pg.mkPen(color = 'g', width=1))
        win.nextRow()
        p3  = win.addPlot()
        p3.plot(signal3, pen = pg.mkPen(color = 'b', width=1))

        if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
                 QtGui.QApplication.instance().exec_()

    def generateFile(self,output, signaltype, file, filenum):
        path = "C:\\Users\\danos\\source\\repos\\WFDB\\WFDB\\%s\\" %(file)
        np.savetxt(path+"%s_%s_%s.csv" %(signaltype,file,filenum), output, delimiter=",") 
        
    
class Records:
    if __name__ == '__main__':  
        ecg = np.array([])
        ip = np.array([])
        spo2 = np.array([])
        fs =0      
        file  = input("Which record to read?")
        fileNum = input("which file you want to read?")
        if fileNum<9:
            filenum= "000"+str(fileNum)
        else:
            filenum="00"+str(fileNum)

        path = "%s\\%s_%s"  %(file,file,filenum)
        print path
        try:
            print filenum
            record  = wfdb.rdsamp(path, channels = [0,1,2])
            print "sample Read"
            Signal1 = record.p_signals
            print Signal1
            ecg = record.p_signals[:,0]
            print "ECG:", ecg
            ip = record.p_signals[:,1]
            print "IP",ip
            spo2 = record.p_signals[:,2]
            print  "SPO2", spo2
            fs = record.fs
            print "SampleFrequency",fs


            plotter = graficador()
            plotter.plotSignals(ecg,ip,spo2)    

        except:
                print "algo paso mal"
