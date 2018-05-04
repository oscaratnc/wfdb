import wfdb
import os 


if __name__ == '__main__': 

    recordsPath = []
    recordID =[]
    #fh  = open("recordsNeonates.txt")

    #for line in fh:
        #line = line.rstrip()
    line = "30/3001676"
    recordsPath.append('mimic3wdb/'+line)
    record  = line.split('/')
    recordID.append(record[1])

    #for i  in range (len(recordsPath)):
    print recordID
    print recordsPath
    dir  = os.getcwd()
    print dir 
    directory = dir + '\\'+recordID[0]
    
    print recordsPath[0]
    try:
        print "downloading", recordsPath[0]
        wfdb.dl_database(recordsPath[0],directory)
        
    except:
            print "algo paso :("
    #i += 1 
       
       
    

