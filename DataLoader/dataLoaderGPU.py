


import cudf

def getPreprocessedDataOnGPU():
    
    #Concatenate positive and negative samples
    trainFull = cudf.read_csv("Data/trainFull.csv", index_col = 0)
    testFull = cudf.read_csv("Data/testFull.csv", index_col = 0)
    
    #Randomize samples
    trainFull = trainFull.sample(len(trainFull), random_state=42)  
    testFull = testFull.sample(len(testFull), random_state=42) 

    #Reduce features and extract labels
    trainX = trainFull.iloc[:,:-1].astype('float32')
    trainY = trainFull.iloc[:,-1].astype('int32')
    testX = testFull.iloc[:,:-1].astype('float32')
    testY = testFull.iloc[:,-1].astype('float32')
    

    return trainX, trainY, testX, testY