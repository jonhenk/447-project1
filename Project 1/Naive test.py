import numpy as np
import pandas as pd
import os

#def readData(path):
#    data = pd.read_csv(path, index_col=0)
#    return data

#def classInfo(data):
#    Classes = data.Class.unique()
#    No_C = []
#    for n in Classes:
#        count = data['Class'].value_counts()[n]
#        No_C.append([n,count])
#    return No_C
#data = readData('Data/breast-cancer-wisconsin.data')
#No_C = classInfo(data)
#print(No_C)
#print("done, class test")

class NaiveBayes:
    def __init__(self, path):
        self.df = pd.read_csv(path, index_col=0,header=0) # imports .data to dataframe
        self.classes = self.df['Y'].unique() # creates list of classes / Y / outputs
    def get_classes(self):
        No_C = []
        for n in self.classes:  # iterates through classes and appends proportion to the list
            count = self.df['Y'].value_counts()[n] # total occurrences of class
            No_C.append([n, count/self.df.shape[0]]) # append to list
        return No_C
    def train(self):
        pass

#bc = NaiveBayes('Data/breast-cancer-wisconsin.data')
#print(bc.get_classes())

files = [] # list of .data files
for file in os.listdir("Data"): # goes through files in /Data/ folder
    if file.endswith('.data') and not file.endswith('scrambled.data'): # selects .data files avoids scrambled files for now
        cF = NaiveBayes('Data/'+file) #initializes Naive Bayes with current file
        print(cF.get_classes()) # gets classes and classes count for file and prints
        files.append([file,cF.get_classes()]) # adds classes and classes counts to list

print(files)