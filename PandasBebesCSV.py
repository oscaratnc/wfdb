import glob
import pandas as pd
import graficador as gph
import wfdb
import sqlite3 as sq

class DataFrametoDB:
    def readNDataFiles(self):
        path = r'C:\\Users\\danos\\source\\repos\\WFDB\\WFDB\\GlobalMeasures\\'
        globalMeasures = glob.glob(path+"\\*.hea")
        for i in range( len( globalMeasures ) ):
            print i, globalMeasures[i]

        selectedMeasure = input("Select a Measure to read:")
        print globalMeasures[selectedMeasure]


        SMsplit = globalMeasures[selectedMeasure].split("\\")
        SMsplit =  SMsplit[-1].split('.')
        print SMsplit[0]


        record = wfdb.rdrecord('GlobalMeasures\\'+SMsplit[0])
        data = record.p_signal
        nDF = pd.DataFrame(data,columns=record.sig_name)

        nDF = nDF.fillna(method= 'ffill')
        nDF = nDF.dropna(axis = 0, how= 'any')

        nDF = nDF.drop(['NBP','NBP Sys','NBP Dias', 'NBP Mean'], axis = 1)
        nDF= nDF.rename(index =str, columns = {'RESP':'RR'})

        print nDF.shape

        print "Connecting...."
        con = sq.connect( 'C:\\Users\\danos\\source\\repos\\CETesis\\BabyRecords.db')
        cur = con.cursor()
        print "Connected!"
        tableName = 'RN'+SMsplit[0]
        query = "DROP TABLE IF EXISTS "+ tableName
        cur.execute(query)
        print "Beginning to insert"

        query = 'CREATE TABLE '+tableName + '(ID INTEGER PRIMARY KEY AUTOINCREMENT, HR REAL NOT NULL, PULSE REAL NOT NULL, RR REAL NOT NULL, SpO2 REAL NOT NULL)'
        print query
        cur.execute(query)
        nDF.to_sql(tableName,con,if_exists ='append', index = False )
        print "Finished"

    def jointDataFiles(self):

        path = r'C:\\Users\\danos\\source\\repos\\WFDB\\WFDB\\'

        folderFiles = glob.glob(path+ "**")
        for i in range(len(folderFiles)):
            print i, folderFiles[i]
        folder = input("Select Folder to concatenate:")
        print folderFiles[folder]

        selectedFolder = glob.glob(folderFiles[folder]+"\\*.hea")

        filestoRead = selectedFolder[1:len(selectedFolder)-1]
        fileSplit  = filestoRead[0].split("\\")
        folderNumber = fileSplit[13]
        filenumberList = []

        for  i in range(len(filestoRead)):
            fileSplitNum = filestoRead[i].split("\\")
            filenumber = fileSplitNum[14].split('_')
            filenumber = filenumber[1].split('.')
            filenumber = filenumber[0]
            filenumberList.append(filenumber)

        print folderNumber
        print filenumber
        print filenumberList
        print "Connecting...."
        con = sq.connect( 'C:\\Users\\danos\\source\\repos\\CETesis\\BabyRecords.db')
        cur = con.cursor()
        print "Connected!"


        # tableName = 'RN'+folderNumber
        # query = "DROP TABLE IF EXISTS "+ tableName
        # cur.execute(query)
        # query = "CREATE TABLE  %s ( ID INTEGER PRIMARY KEY AUTOINCREMENT,AVR REAL NOT NULL, ECG REAL NOT NULL ,I REAL NOT NULL, III REAL NOT NULL, PLETH REAL NOT NULL , RESP REAL  NOT NULL, V REAL NOT NULL)" %(tableName)
        # print query
        # cur. execute(query)

        for j in range(len(filestoRead)):
            df = pd.DataFrame()
            record  = wfdb.rdrecord("%s\\%s_%s"  %(folderNumber,folderNumber,filenumberList[j]))

            data = record.p_signal
         

            df = pd.DataFrame(data,columns=record.sig_name)
            df = df.fillna(method = 'ffill')
            df = df.dropna(axis = 0, how = 'any')
            df= df.rename(index = str, columns = {'II':'ECG'})
            
           
            # columnsExpected = ['AVR','ECG','I','III','PLETH','RESP','V']

            # for d in columnsExpected:

            #     if d not in df.columns:
            #         df[d] = 0

            df = df.sort_index(axis = 1)
            babyDFColumns = df.columns.values
            print babyDFColumns
            print df.shape

            # print "Beginning to insert"
            # df.to_sql(tableName,con,if_exists ='append', index = False)
            # print "df", j, "Added"




    def getPossibleApnoeas(self):
        measureFile = ["3001200n", "3001200"]
        con = sq.connect( 'C:\\Users\\danos\\source\\repos\\CETesis\\BabyRecords.db')
        path = r'C:\\Users\\danos\\source\\repos\\CETesis'
        print "Path: ", path
        csvpath = path +"\\\\"+measureFile[1]+"lowsat.csv"
        print "CSV Path: ", csvpath
        vsMonitorMeasureDF = pd.read_sql_query("SELECT * from RN"+measureFile[0]+" WHERE SpO2 <90 AND SpO2 > 0 AND HR>0 AND RR>0 ", con)
        try:
            vsMonitorMeasureDF.to_csv(csvpath)
            print "File Generated"
        except:
            print "Error "

    def generateGraphapnoea(self):
        tableName = ['RN3001200n', 'RN3001200']
        infTime = input ("Initial Time: ")
        supTime = input ("Final Time: ")

        sampleFreq = 125.

        infTimeFreq = (infTime+2) * sampleFreq
        supTimeFreq = (supTime+2) * sampleFreq

        con = sq.connect( 'C:\\Users\\danos\\source\\repos\\CETesis\\BabyRecords.db')
        vsMonitorMeasureDF = pd.read_sql_query("sELECT ID, HR, PUlse, rr, Spo2 from %s where ID<%s and ID >%s" %( tableName[0], str(supTime), str(infTime) ),con)
        allMeasureDF = pd.read_sql_query("select ID, ECG, RESP, PLETH from %s where ID< %s and ID> %s" %(tableName[1], supTimeFreq, infTimeFreq),con)
        heartRate = vsMonitorMeasureDF['HR'].values
        PulseRate = vsMonitorMeasureDF['PULSE'].values
        respRate = vsMonitorMeasureDF['RR'].values
        Spo2 = vsMonitorMeasureDF['SpO2'].values
        Resp = allMeasureDF['RESP'].values
        ecg = allMeasureDF['ECG'].values
        pleth = allMeasureDF['PLETH'].values

        graph = gph.Graficador()
        graph.plotSignals( infTime, supTime, Spo2,heartRate,PulseRate,respRate,ecg,Resp,pleth)



class main:
    csvF = DataFrametoDB()
    csvF.jointDataFiles()
    csvF.readNDataFiles()
    csvF.getPossibleApnoeas()
    # #csvF.generateGraphapnoea()