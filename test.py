
import wfdb
import pandas  as pd
import glob


path = r'C:\\Users\\danos\\source\\repos\\WFDB\\WFDB\\'

folderFiles = glob.glob(path+ "**")
for i in range(len(folderFiles)):
    print i, folderFiles[i]
folder = input("Select Folder to concatenate:")
print folderFiles[folder]

selectedFolder = glob.glob(folderFiles[folder]+"\\*.hea")
csvNum = 0

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



framesList= []

for j in range(len(filestoRead)):
    tempDF = pd.DataFrame()
    record  = wfdb.rdrecord("%s\\%s_%s"  %(folderNumber,folderNumber,filenumberList[j]))
    
    data = record.p_signal
    df = pd.DataFrame(data,columns=record.sig_name)
    
    if 'V' in df.columns:
        df = df.drop(columns = 'V')
    if 'II' in df.columns:
        df = df.drop( columns='II')
    
    if 'PLETH' not in df.columns:
        df['PLETH'] = "0" 
    
    df = df.sort_index(axis = 1)
    framesList.append(df)

babyDF = pd.concat(framesList)
print "concat Done"
print babyDF.shape

babyDF.to_csv(folderFiles[folder]+"\\AllMeasures.csv",index = False)