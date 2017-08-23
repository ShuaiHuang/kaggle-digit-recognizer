import csv
import numpy as np

def processDataset(trainDatasetFilePath, testDatasetFilePath):
    trainDataset=getDataset(trainDatasetFilePath, True)
    testDataset=getDataset(testDatasetFilePath, False)

def getDataset(filePath, containsLabel):
    with open(filePath, 'r') as fileHandler:
        datasetReader=csv.reader(fileHandler)
        for line in datasetReader:
            pass # TODO: Think twice for the formation of serialized data.