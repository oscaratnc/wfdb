from scipy import signal as sp
from scipy import interpolate as ip
import numpy as np
import matplotlib.pyplot as plt


class globalV:
    cleanECG = np.array([])
    cleanECGNormalized = np.array([])
    cleanECGBLD = np.array([])
    cleanECGFilteredLP = np.array([])
    cleanECGDerivative  = np.array([])
    cleanECGSquared = np.array([])
    cleanECGintegrated = np.array([],np.float)
    HR_vector = np.array([],np.float)
    ecgRRDuration_vector= np.array([],np.float)
    thresholdECG = np.array([])

    fs = 0
    spO2SignalFilter = np.array([])
    spO2SlopeSum = np.array([])
    thresholdSpO2 = 0
    
    q_peak= []  
    r_peak= []
    s_peak= []
    q_Loc = []
    r_Loc = []
    s_Loc = []
    
    edrRS = []
    edrRSFiltered = []
    rsaEDR = []
    areaEDR = []
class generalProcessing:

    def movMean (self, signal, window):
        WinFilt = np.repeat(1.0,window)/window
        meanfilSignal = np.convolve(signal,WinFilt,'valid')
        return meanfilSignal

    def BPButterFilter(self,signal,flow,fhigh,sampleF,order):
        sampleRate = sampleF
        nyq_rate = sampleRate/2.0
        Wn = [flow/nyq_rate,fhigh/nyq_rate]
        n = order
        [b,a] = sp.butter(n,Wn,'bandpass')
        filtered = sp.filtfilt(b,a,signal)
        return filtered

    def smooth(self,x, window_size):
        box = np.ones(window_size)/window_size
        return np.convolve(x, box, mode='same')

    def getACcomponent(self, measure):
        mean = np.mean(measure)
        measure = (measure-mean)
        return measure

    def removeNAN(self, Signal):
        Signal= np.nan_to_num(Signal)
        print "nan values removed"
        return Signal

    def getDCComponent(self,measure):
        DCcomponent  = np.mean(measure)
        return DCcomponent

    def Normalize(self, measure):
        absS = np.max(np.abs(measure))
        measureN = measure/absS
        measureN = np.round(measureN,4)
        return measureN

class ECGProcessing:

    def eraseBLD(self, ECGSignal, sampleFreq):
        #print "200ms Window"
        #Number of 200ms Window samples  calculation
        n = int(200*sampleFreq/1000)
        line1 = np.array([])
        line2 = np.array([])


        for k in range (len(ECGSignal)):
            line0 = np.zeros(n)


            lim1 = k - (n/2)
            lim2 = k + (n/2) - 1

            if lim2 > len(ECGSignal):
                lim2 = len(ECGSignal)

            if lim1 <= 0:
                for k2 in range((n/2)+lim1,lim2):
                    line0[k2]   = ECGSignal[k]
                    k = k + 1
            else:
                k1 = 0
                for k2 in range(lim1,lim2):
                    line0[k1] = ECGSignal[k2]
                    k1 = k1+1


            line0 = np.sort(line0)

            if lim1 <= 0:
                r1 = n+lim1-2
                r2 = n+lim1-1
                prom = (line0[r1]+line0[r2])/2

            else:

                prom = (line0[n/2]+line0[n/2 + 1])/2

            line1 = np.append(line1,prom)


        #print "200 ms Window Finished"
        #600ms Window
        #Number of 600ms Window Samples Calculation
        n = int((600 *sampleFreq)/1000)

        for  k in range (len(ECGSignal)):
            line0 = np.zeros(n)
            lim1 = k - (n/2)
            lim2 = k + (n/2)-1

            if lim2 > len(ECGSignal):
                lim2 = len(ECGSignal)

            if lim1 <= 0:
                for k2 in range(((n/2)+lim1), lim2):
                    line0[k2] = line1[k]
                    k=k+1
            else:

                k1 = 0
                for k2 in range(lim1, lim2):
                    line0[k1]  = line1[k2]
                    k1 = k1 + 1


            line0 = np.sort(line0)

            if lim1 <= 0:

                prom = (line0[n+ lim1 -2]+line0[n+lim1-1])/2

            else:

                prom = (line0[n/2]+line0[n/2 + 1])/2



            line2 = np.append(line2,prom)


        #print "600ms Window Finished",
        #print "signal: ", len(ECGSignal)
        #print "line0: ", len(line0)
        #print "line1: ", len(line1)
        #print  "line2: ", len(line2)

        return ECGSignal-line2

    def eraseBLD2(self, ECGSignal, fs):
        nyq  =  fs * 0.5
        f1 =.4
        beta= 4
        b= sp.firwin(20,f1,window=('kaiser', beta),)
        ECGSignal= sp.filtfilt(b,1,ECGSignal)
        return ECGSignal

    def digitalLPFilterECG(self, ECGSignal,sampleFreq):
        nyq = sampleFreq * 0.5
        f2 = (40.0/nyq)
        b,a= sp.butter(1,f2)
        signalFiltered = sp.filtfilt(b,a,ECGSignal)

        gp = generalProcessing()
        signalFiltered= gp.Normalize(signalFiltered)
        return signalFiltered

    def digitalHPFilterECG(self, ECGSignal,sampleFreq):
        nyq = sampleFreq * 0.5
        f1 =.5/nyq
        b = sp.firwin(4, f1)
        signalFiltered = sp.filtfilt(b,1,ECGSignal)

        gp = generalProcessing()
        signalFiltered= gp.Normalize(signalFiltered)
        return signalFiltered

    def bandpassFilterECG (self, ECGsignal, sampleFreq):
        ECGsignal  = self.digitalLPFilterECG(ECGsignal,sampleFreq)
        ECGsignal  = self.digitalHPFilterECG(ECGsignal, sampleFreq)
        ECGsignal  = ECGsignal / np.max(np.abs(ECGsignal))
        return ECGsignal

    def derivativeFilter (self, ECGSignal):
        ECGSignaldF = sp.savgol_filter(ECGSignal, 5,2,deriv=1)
        ECGSignaldF=ECGSignaldF / np.max(np.abs(ECGSignaldF))
        return ECGSignaldF

    def squaringSignal (self, Signal):
        squaringSignal= np.array([0 for i in range(len(Signal))],dtype = np.float)
        for i in range(len(Signal)):
            signalSquared = Signal[i]**2
            squaringSignal[i]  = float(signalSquared)
        squaringSignal = squaringSignal / np.max(np.abs(squaringSignal))
        return squaringSignal

    def tempWinIntegration(self, ecgSignal):
        h = np.ones(15)/15
        delay = 5
        IntFilter = sp.convolve(ecgSignal,h)
        IntFilter = IntFilter[delay:len(ecgSignal)+delay]
        gp = generalProcessing()
        gp.movMean(IntFilter,3)
        IntFilter = IntFilter / np.max(np.abs(IntFilter))
        return IntFilter

    def cleanECG(self, ECGSignal, sampleFreq):
        gP = generalProcessing()

        #DC Component Elimination
        globalV.cleanECGAC = gP.getACcomponent(ECGSignal)

        #Normalize ECG Signal after DC component Removal
        globalV.cleanECGNormalized = gP.Normalize(globalV.cleanECGAC)

        #Baseline Drift correction
        globalV.cleanECGBLD = self.eraseBLD(globalV.cleanECGNormalized,sampleFreq)

        #BandPassFilter
        globalV.cleanECGFilteredLP = self.bandpassFilterECG(globalV.cleanECGBLD,sampleFreq)
        #globalV.cleanECGFilteredLP = gP.BPButterFilter(globalV.cleanECGBLD,5,15,sampleFreq, 20)

        #Derivative Filter
        globalV.cleanECGDerivative = self.derivativeFilter(globalV.cleanECGFilteredLP)

        #squaring function
        globalV.cleanECGSquared = self.squaringSignal(globalV.cleanECGDerivative)

        #integrating function
        globalV.cleanECGintegrated = self.tempWinIntegration(globalV.cleanECGSquared)

        # plt.figure()
        # plt.subplot(411)
        # plt.plot(globalV.cleanECGBLD)
        # plt.ylabel('clean BLD')
        # plt.subplot(412)
        # plt.plot(globalV.cleanECGFilteredLP)
        # plt.ylabel('BP Filtered')
        # plt.subplot(413)
        # plt.plot(globalV.cleanECGDerivative)
        # plt.ylabel('Derivative')
        # plt.subplot(414)
        # plt.plot(globalV.cleanECGSquared)
        # plt.ylabel('Squared')

        # plt.figure()
        # plt.plot(globalV.cleanECGintegrated)
        # plt.plot(globalV.cleanECGFilteredLP)
        # plt.ylabel('Integrated & Filter')
        # plt.show()


        #print "\n ECG Cleaned"

    def getMax(self,signal):
        peakVal = np.max(signal)
        peakLoc = np.argmax(signal)
        return peakVal, peakLoc

    def getMin(self,signal):
        peakVal = np.min(signal)
        peakLoc = np.argmin(signal)
        return peakVal, peakLoc

    def  peakLocation(self):
        ECGSignal = globalV.cleanECGintegrated

        N = len(ECGSignal)

        max_h = np.max(globalV.cleanECGintegrated)
        thresh = np.mean(globalV.cleanECGintegrated)

        poss_reg = [0]*N
        poss_regV = [0]*N

        globalV.thresholdECG = np.ones(len(ECGSignal)) *(thresh* max_h)

        poss_regV = ECGSignal > thresh*max_h
        #print poss_regV

        for i in range(len(poss_reg)):
            if poss_regV[i] == True:
                poss_reg[i] = 1
            else:
                poss_reg[i] = 0

        #print poss_reg

        #print "diffLeft", diffLeft
        #print "diffRight", diffRight

        left =np.transpose(np.nonzero(np.diff(poss_reg) == 1))
        right = np.transpose(np.nonzero(np.diff(poss_reg) == -1))


        # left = left[:len(left)]
        # right= right[1:len(right)]

        # 
        # temp = right

        # for i in range (len(left)):
        #     print"left ", left[i], " right: ", right[i]
        k = 0
        ecg=  globalV.cleanECGFilteredLP
        #print "RP",right
        Q_value = np.array([])
        Q_loc = np.array([])
        R_value = np.array([])
        R_loc = np.array([])
        S_value = np.array([])
        S_loc = np.array([])

        if  left[0] < right[0]:
                left = left[1:]

        if right[0] < left[0]:
            right  = right[1:]

        # print "left: ",len(left),
        # print left
        # print "right: ", len (right)
        # print right

        for iLoc in range(len(left)-1):


            # print "right:", len(right), right
            # print "left: ", len(left), left
            # print  "iLoc: ", iLoc


            
            # #For normal QRS Complexes
            # print "Right loc:", right[iLoc], "Left loc: ", left[iLoc], "Diff", right[iLoc]-left[iLoc]


            if right[iLoc]-left[iLoc] > 5:
                
                #Stores ECG values in temporal vector
                tempECGR = ecg[int(left[iLoc]):int(right[iLoc])]
                # plt.figure()
                # plt.subplot(311)
                # plt.title("TempECGR")
                # plt.plot(tempECGR)
               
                #print  "tempECGR", tempECGR

                #calculate the max peak of the temporal vector
                peakValR, peakLocR = self.getMax(tempECGR)
               
                # print peakLocR
                # print int(left[iLoc])

                tempECGQ = ecg[int(left[iLoc]):int(left[iLoc])+peakLocR]
                # plt.subplot(312)

                # plt.title("TempECGQ")
                # plt.plot(tempECGQ)
                
                tempECGS = ecg[int(left[iLoc])+peakLocR:int(right[iLoc])]
                # print "tempECGS", tempECGS
                # plt.subplot(313)
                # plt.title("TempECGS")
                # plt.plot(tempECGS)
                # plt.show()
                if peakValR < 0:
 
                    #calculate the min peak of the temporal vector
                    peakValR, peakLocR = self.getMin(tempECGR)
                    R_loc = np.append(R_loc, peakLocR + left[iLoc])
                    R_value = np.append(R_value, peakValR)
                    
                    peakValQ, peakLocQ = self.getMax(tempECGQ)
                    Q_loc = np.append(Q_loc, (peakLocQ) + left[iLoc])
                    Q_value = np.append(Q_value, peakValQ) 

                    peakValS, peakLocS = self.getMax(tempECGS)
                    S_loc = np.append(S_loc, (peakLocS) + R_loc[iLoc])
                    S_value = np.append(S_value, peakValS) 

                    k = k + 1
                    continue
                R_loc = np.append(R_loc, (peakLocR) + left[iLoc])
                R_value = np.append(R_value, peakValR)

                peakValQ, peakLocQ = self.getMin(tempECGQ)

                Q_loc = np.append(Q_loc, (peakLocQ) + left[iLoc])
                Q_value = np.append(Q_value, peakValQ) 

                peakValS, peakLocS = self.getMin(tempECGS)
                S_loc = np.append(S_loc, peakLocS + R_loc[k])
                S_value = np.append(S_value, peakValS) 

               
                k = k +1
        #print "R localtions ", R_loc[90:93]
        
        globalV.r_Loc = R_loc
        globalV.r_peak = R_value
        globalV.q_Loc = Q_loc
        globalV.q_peak= Q_value
        globalV.s_Loc = S_loc
        globalV.s_peak = S_value

    def HRcalculation(self,ECGSignal,fs):
        #cleans ECG and fills global values
        self.cleanECG(ECGSignal,fs)

        #fills R_Loc global value
        self.peakLocation()

        #Calculates  Heart Rate
        if len(globalV.r_Loc) > 3:
            t = np.arange(0,len(globalV.cleanECGFilteredLP),1./fs)
            RR = np.array([])
            for x in range(len(globalV.r_Loc)-1):
                RR_t= t[int(globalV.r_Loc[x+1])]- t[int(globalV.r_Loc[x])]
                RR = np.append(RR,RR_t)
            #print RR
            meanRR = np.mean(RR)
            #print "MEAN RR: ", meanRR
            HR = 60/meanRR
            #print "HR: ", HR
            globalV.HR_vector = np.append(globalV.HR_vector,HR)
            #print "HR ADDED"

class spo2Processing:

    def cleanSpo2Sginal(self, IRsignal, RedSignal, sampleRate):
        gp = generalProcessing()
        #get Red and Ir buffers
        irSignal= IRsignal
        redSignal = RedSignal

        # #Normalize Red and IR signals
        redSignal = gp.Normalize(redSignal)
        irSignal= gp.Normalize(irSignal)

        #Butterword 4th order bandpass filter .5-6Hz
        irSignal = gp.BPButterFilter(irSignal, 0.5, 4.0,sampleRate,4)
        redSignal = gp.BPButterFilter(redSignal, 0.5, 4.0,sampleRate,4)

        #  #fft filtered dignal
        # self.IR_Filtered_FFT = spfft.fft(self.IR)
        # self.Red_Filtered_FFT= spfft.fft(self.Red)

        #Mean filter widnow = 4
        # self.IR  = pro.movMean(self.IR,4)
        # self.Red = pro.movMean(self.Red,4)
        redSignal = sp.medfilt(redSignal,3) *- 1
        irSignal = sp.medfilt(irSignal,3) * -1

        return redSignal,irSignal

    def spO2LowPassFilter(self, signal):
        #Digital  Low Pass Butterwort Digital  Filter at 16Hz
        nyq = 125./2
        b, a = sp.butter(N=2,Wn =[2/nyq,16/nyq],btype='bandpass', analog = False)
        filtered =  sp.filtfilt(b,a,signal)
        #print filtered
        return filtered

    def slopeSum(self, signal, sampleFreq):
        print "Filtering in process...."
        globalV.spO2SignalFilter = self.spO2LowPassFilter(signal)
        print globalV.spO2SignalFilter
        print "Slope Sum in process"
        gP = generalProcessing()
        window = (128 * sampleFreq)/1000
        slopeSum = np.array([])
        #signal = gP.getACcomponent(signal)
        for k in range(len(signal)):
            #print "k = ", k
            extractedWindow = np.zeros(window)


            #limit declaration for extraction
            lim1 = k  - (window/2)
            lim2 = k + (window/2) -1


            if lim2 > len(signal):
                lim2 = len(signal)
            if lim1 < 0:
                for n in range((window/2)+lim1,lim2):
                    #print signal[k]
                    extractedWindow[n] = signal[k]
                    #print "ex: ", extractedWindow[n]
                    k = k+1
            else:
                t = 0
                for x in range(lim1, lim2):
                    #print signal[x]
                    extractedWindow[t] = signal[x]
                    t = t+1

            slope = 0
            #print extractedWindow
            for y in range (1,len(extractedWindow)):

                slopeTemp = extractedWindow[y]- extractedWindow[y-1]
                #print "w1 = ",extractedWindow[y], "w2= ",extractedWindow[y-1], "sl= ", slope
                if slopeTemp <= 0:
                    slopeTemp = 0

                slope = slopeTemp + slope
            #print slope
            slopeSum = np.append(slopeSum, slope)
        return slopeSum

    def HRcalculation(self, signal):
        N = len(signal)
        peaksLoc= np.array([])

        globalV.spO2SlopeSum = self.slopeSum(signal)
        #print globalV.spO2SlopeSum
        globalV.thresholdSpO2 = .5*(3*np.mean(globalV.spO2SlopeSum))

        poss_reg = [0]*N
        poss_regV  = globalV.spO2SlopeSum > globalV.thresholdSpO2

        for i in range(len(poss_regV)):
            if poss_regV[i] == True:
                poss_reg[i] = 1
            else:
                poss_reg[i] = 0

        # for x in poss_reg:
        #             print x

        left = np.transpose(np.nonzero(np.diff(poss_reg) == 1))
        right = np.transpose(np.nonzero(np.diff(poss_reg)== -1))

        #print "left \n", left
        #print  " right:\n", right

        for iLoc in range(len(left)-1):
            temporal = []

            if right[iLoc]-left[iLoc] > 1 :

                for iRL in range(left[iLoc], right[iLoc]):
                    temporal.append(globalV.spO2SignalFilter[iRL])
                #print temporal
                peakLoc = np.argmax(temporal)
                #print peakLoc, temporal[peakLoc]
                peaksLoc = np.append(peaksLoc, peakLoc+left[iLoc])

        #print peaksLoc
        if len(peaksLoc) > 1:
            t = np.arange(0, len(peaksLoc), 1./125)
            RR = np.array([])

            for n in range(1, len(peaksLoc)-1):
                RR_t = t[int(peaksLoc[n+1])]-t[int(peaksLoc[n])]
                RR = np.append(RR,RR_t)

            meanRR = np.mean(RR)
            #print "Mean RR: ", meanRR

            pulse = 60/meanRR
            print pulse
        return pulse

    def calcSpO2(self,signalRed, signalIR):
        gP  = generalProcessing()
        DCRed = gP.getDCComponent(signalRed)
        acRed = gP.getACcomponent(signalRed)
        DCIR = gP. getDCComponent(signalIR)
        acIR = gP.getACcomponent(signalIR)

        RR =round(np.mean((acRed/DCRed)/(acIR/DCIR)),4)
        spO2Value =  96.545 + 0.616 * RR
        Spo2Value =int(np.round(spO2Value,0))

class EDRcalculation:


    def interpolation(self, signal,freq, derivative=0 ):
        x = np.arange(0,len(signal))
        signal = ip.CubicSpline(x,signal)
        if derivative <1:
            signal = signal(x)
        else:
            signal = signal(x,derivative)

        
        gP = generalProcessing()
        nyquist = freq * 0.5
        f1 = 4
        b= sp.firwin(5, f1, nyq=nyquist)
        signal= sp.filtfilt(b,1,signal)
        samples = int(len(signal)*(100./freq))
        signal = sp.resample(signal,samples)
        signal = gP.Normalize(signal)

        return signal
        
    def RS_AmplitudeEDR(self):
        r_loc = globalV.r_Loc
        #r_loc500 = []
        rPeakValues = globalV.r_peak
        sPeakValues = globalV.s_peak
        rsValue = []

        # for x in r_loc:
        #     r_loc500.append(x*(500/125))
        # #print r_loc500
        
        hRV= np.zeros(len(globalV.cleanECGFilteredLP))

        for peakVal in range (len(rPeakValues)):
            ampR = rPeakValues[peakVal]
            ampS = sPeakValues[peakVal]           
            #rsValue.append(ampR-ampS)
            rsValue.append(abs(ampR) - abs(ampS))
        
        #print len(r_loc500), len(hRV), len(rsValue)
        k = 0
        for  i in range(len(r_loc)): 
            qrsloc = int(r_loc[i])
            #print   i, qrsloc, k
            hRV[qrsloc]= rsValue[k]
            k = k+1
           

        globalV.edrRS = hRV
        nyquist = globalV.fs * 0.5
        f1 = .8
        b= sp.firwin(150, f1, nyq= nyquist)
        globalV.edrRS= sp.filtfilt(b,1,globalV.edrRS)
        globalV.edrRS = self.interpolation(globalV.edrRS[65:len(globalV.edrRS)-65],globalV.fs)
        #globalV.edrRS = globalV.edrRS[50:(len(globalV.edrRS)-50)]    
    def QRSareaEDR(self):
        r_loc = globalV.r_Loc

        qPeakValues = globalV.q_peak
        rPeakValues = globalV.r_peak
        sPeakValues = globalV.s_peak
        areaQRS = []
        hRV= np.zeros(len(globalV.cleanECGFilteredLP))
    
        for i in range(len(globalV.r_Loc)):
            qrsArea = qPeakValues[i] + rPeakValues[i] + sPeakValues[i]
            areaQRS.append(qrsArea)
        k = 0
        for  i in range(len(r_loc)): 
            qrsloc = int(r_loc[i])
            hRV[qrsloc]= areaQRS[k]
            k = k+1
        
        globalV.areaEDR = hRV
        nyquist = globalV.fs * 0.5
        f1 = .8
        b= sp.firwin(200, f1, nyq =nyquist)
        globalV.areaEDR= sp.filtfilt(b,1,globalV.areaEDR)
        globalV.areaEDR = self.interpolation(globalV.areaEDR[65:len(globalV.areaEDR)-65],globalV.fs)
    def RsaEDR(self):
            nyquist = globalV.fs * .5    
            ecgSignal = globalV.cleanECGFilteredLP  
            f1 = [.2, .8]
            b= sp.firwin(500, f1, pass_zero=False, nyq=nyquist)
            globalV.rsaEDR= sp.filtfilt(b,1,ecgSignal)
            
            globalV.rsaEDR = self.interpolation(globalV.rsaEDR, globalV.fs,1)
        
    def EdrCalcResps(self, ECGSignal,fs):
            globalV.fs = fs
            ecgP =  ECGProcessing()
            ecgP.cleanECG(ECGSignal,fs)
            ecgP.peakLocation()
            
            self.RS_AmplitudeEDR()
            self.RsaEDR()
            self.QRSareaEDR()
      
        