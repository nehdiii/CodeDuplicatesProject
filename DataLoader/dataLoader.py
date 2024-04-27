
import pandas as pd

def getPreprocessedData():
    
    #Concatenate positive and negative samples
    trainFull = pd.read_csv("Data/trainFull.csv", index_col = 0)
    testFull = pd.read_csv("Data/testFull.csv", index_col = 0)
    
    #Randomize samples
    trainFull = trainFull.sample(len(trainFull))
    testFull = testFull.sample(len(testFull))

    #Randomize samples
    trainFull = trainFull.sample(len(trainFull))
    testFull = testFull.sample(len(testFull))

    #Reduce features and extract labels
    trainX = trainFull.iloc[:,:-1]
    trainY = trainFull.iloc[:,-1]
    testX = testFull.iloc[:,:-1]
    testY = testFull.iloc[:,-1]

    #Reshape data to 3D for CNN
    trainX = trainX.to_numpy()[..., None]
    trainY = trainY.to_numpy()[..., None]
    testX = testX.to_numpy()[..., None]
    testY = testY.to_numpy()[..., None]

    return trainX, trainY, testX, testY