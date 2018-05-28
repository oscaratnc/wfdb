import matplotlib.pyplot as plt
import numpy as np
import sqlite3 as sq
import Processing
import pandas as pd

print "Connecting to database!...."
con = sq.connect('C:\\Users\\danos\\source\\repos\\CETesis\\BabyRecords.db')
#plethSignal = pd.read_sql_query('SELECT PLETH FROM RN3000775 WHERE ID > 0 AND ID < 125 ', con )
s1= raw_input('From second:')
s2 = raw_input('To second:')
ecgSignal = pd.read_sql_query('SELECT ECG FROM RN3000775 WHERE Time >%s AND Time <%s' %(s1,s2), con )
#respSignal = pd.read_sql_query('SELECT RESP FROM RN3000775 WHERE Time >0 AND Time < 20', con )
deltaS = int(s2)-int(s1)
ecgSignal = ecgSignal['ECG'].values
print "Data obtained"


#plethSignalList = plethSignal['PLETH'].values
#plethSignalList = sp.detrend(plethSignalList)s
#print plethSignalList
#pro = Processing.spo2Processing()
#pro.HRcalculation(plethSignalList)

print "Processing begin...."
fs = 125
pro = Processing.EDRcalculation()
pro.EdrCalcResps(ecgSignal,fs)
print "Processing Finished, graphing....."


# spo2Filtered = Processing.globalV.spO2SignalFilter
# slopesum = Processing.globalV.spO2SlopeSum
# threshold = Processing.globalV.thresholdSpO2




plt.figure("ECG")
plt.subplot(411)
t = np.arange(0,len(Processing.globalV.cleanECGFilteredLP))*(1./fs)
tedrRS = np.arange(0,len(Processing.globalV.edrRS))*(1./100)
trsaRS = np.arange(0,len(Processing.globalV.rsaEDR))*(1./100)
tareaRS = np.arange(0,len(Processing.globalV.areaEDR))*(1./100)
tr_loc = []
ts_loc =[]
tq_loc = []

for x in Processing.globalV.r_Loc:
    tr_loc.append(x /fs)
#print tr_loc
for y in Processing.globalV.s_Loc:
    ts_loc.append(y/fs)

for j in Processing.globalV.q_Loc:
    tq_loc.append(j/fs)

plt.plot(Processing.globalV.cleanECGFilteredLP)
# plt.plot(t, Processing.globalV.cleanECGintegrated)
# plt.plot(t,Processing.globalV.thresholdECG)
# plt.plot(tr_loc,Processing.globalV.r_peak, 'ro')
# plt.plot(ts_loc,Processing.globalV.s_peak, 'go')
# plt.plot(tq_loc, Processing.globalV.q_peak, 'bo')
plt.ylabel("ECG")
plt.xlabel("Time(s)")

#plt.figure("R Amp")
plt.subplot(412)
#plt.plot(t,Processing.globalV.cleanECGFilteredLP)
plt.plot(tedrRS,Processing.globalV.edrRS)
plt.ylabel("RS amp")
plt.xlabel("Time(s)")


#plt.figure("RSA EDR")
plt.subplot(413)
#plt.plot(t,Processing.globalV.cleanECGFilteredLP)
plt.plot(trsaRS,Processing.globalV.rsaEDR)
plt.ylabel("RSA EDR")
plt.xlabel("Time(s)")

plt.subplot(414)
#plt.plot(t,Processing.globalV.cleanECGFilteredLP)
plt.plot(tareaRS,Processing.globalV.areaEDR)
plt.ylabel("Area EDR")
plt.xlabel("Time(s)")

plt.show()




