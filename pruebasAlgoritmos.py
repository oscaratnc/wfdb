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
pro.EdrCalcResps(ecgSignal,deltaS,fs)
print "Processing Finished, graphing....."


# spo2Filtered = Processing.globalV.spO2SignalFilter
# slopesum = Processing.globalV.spO2SlopeSum
# threshold = Processing.globalV.thresholdSpO2




plt.figure(1)
plt.subplot(411)
t = np.arange(0,len(Processing.globalV.cleanECGFilteredLP))*(1./fs)
tr_loc = []
ts_loc =[]
tq_loc = []

for x in Processing.globalV.r_Loc:
    tr_loc.append(x /fs)

for y in Processing.globalV.s_Loc:
    ts_loc.append(y/fs)

for j in Processing.globalV.q_Loc:
    tq_loc.append(j/fs)

plt.plot(t,Processing.globalV.cleanECGFilteredLP,'k')
plt.plot(tr_loc,Processing.globalV.r_peak, 'ro')
plt.plot(ts_loc,Processing.globalV.s_peak, 'go')
plt.plot(tq_loc, Processing.globalV.q_peak, 'bo')

plt.subplot(412)
#plt.plot(Processing.globalV.rsaEDR)
plt.plot(Processing.globalV.edrRS)

plt.subplot(413)
plt.plot(Processing.globalV.rsaEDR)
#plt.plot(range(0,len(spo2Filtered)),spo2Filtered,range(0,len(slopesum)),slopesum,range(0,len(slopesum)), [threshold]*len(spo2Filtered))

plt.subplot(414)
plt.plot(Processing.globalV.areaEDR)
plt.show()




