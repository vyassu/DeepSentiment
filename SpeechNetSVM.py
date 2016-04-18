import numpy as np
import pandas as pd
import datetime
import analyse as an
import glob as gb
import scipy.io.wavfile as sc
import SpeechPitchExtraction as lp
from sklearn import svm
from sklearn import cross_validation
from sklearn.multiclass import OneVsRestClassifier

class speechLSTM:
    # Initializing the LSTM Model
    def __init__(self):
       self.prevData = 100
       self.batchsize=200
       self.model = OneVsRestClassifier(svm.SVC(kernel='poly',gamma=1,C = 1,tol=0.0001,cache_size=5000)  )     #self.model = OneVsRestClassifier(LinearSVC(random_state=0))


    def load_data_file(self):
        outputdata = []
        for f in gb.glob("/media/vyassu/OS/Users/vyas/Documents/Assigments/BigData/AudioData/DC/*.wav"):
            frate, inputdata = sc.read(f)
            pitch=lp.getPitch(f)
            emotion = ""
            loudness = abs(an.loudness(inputdata))
            filename = f.split("/")[-1].split(".")[0]
            if filename[0] == "s":
                emotion = filename[0:2]
                ##emotion = float(int(hashlib.md5(emotion).hexdigest(), 16))
            else:
                emotion = filename[0]
                ##emotion =  float(int(hashlib.md5(emotion).hexdigest(), 16))
            outputdata.append(list([loudness,pitch, emotion]))
        for f in gb.glob("/media/vyassu/OS/Users/vyas/Documents/Assigments/BigData/AudioData/JE/*.wav"):
            frate, inputdata = sc.read(f)
            pitch = lp.getPitch(f)
            emotion = ""
            loudness = abs(an.loudness(inputdata))
            filename = f.split("/")[-1].split(".")[0]
            if filename[0] == "s":
                emotion = filename[0:2]
                ##emotion = float(int(hashlib.md5(emotion).hexdigest(), 16))
            else:
                emotion = filename[0]
                ##emotion = float(int(hashlib.md5(emotion).hexdigest(), 16))
            outputdata.append(list([loudness, pitch, emotion]))
        for f in gb.glob("/media/vyassu/OS/Users/vyas/Documents/Assigments/BigData/AudioData/JK/*.wav"):
            frate, inputdata = sc.read(f)
            pitch = lp.getPitch(f)
            emotion = ""
            loudness = abs(an.loudness(inputdata))
            filename = f.split("/")[-1].split(".")[0]
            if filename[0] == "s":
                emotion = filename[0:2]
                ##emotion = float(int(hashlib.md5(emotion).hexdigest(), 16))
            else:
                emotion = filename[0]
                ##emotion = float(int(hashlib.md5(emotion).hexdigest(), 16))
            outputdata.append(list([loudness, pitch, emotion]))
        for f in gb.glob("/media/vyassu/OS/Users/vyas/Documents/Assigments/BigData/AudioData/KL/*.wav"):
            frate, inputdata = sc.read(f)
            pitch = lp.getPitch(f)
            emotion = ""
            loudness = abs(an.loudness(inputdata))
            filename = f.split("/")[-1].split(".")[0]
            if filename[0] == "s":
                emotion = filename[0:2]
                ##emotion = float(int(hashlib.md5(emotion).hexdigest(), 16))
            else:
                emotion = filename[0]
                ##emotion = float(int(hashlib.md5(emotion).hexdigest(), 16))
            outputdata.append(list([loudness, pitch, emotion]))
        return outputdata

    def get_train_test_data(self,data,percent_split):
        noOfSamples = len(data)*(1-percent_split)
        print("No of Samples", noOfSamples)
        test =  data.iloc[0:int(noOfSamples), 2:]
        test1=[]
        for i in range(len(test)):
            test1 = np.append(test1,test.iloc[i].values[0])

        return data.iloc[int(noOfSamples):, 0:2], data.iloc[int(noOfSamples):, 2:],data.iloc[0:int(noOfSamples), 0:2],np.array(test1)

    def trainNNet(self,data_,label_):
        #data = data_/data_.max(axis=0)
        #label = label_/label_.max(axis=0)
        data=data_
        label=label_
        X_train, X_test, y_train, y_test = cross_validation.train_test_split(data, label.astype(str), test_size = 0.045, random_state = 0)
        self.model.fit(X_train,y_train)
        print("score",self.model.score(X_test,y_test))
        #print (cross_validation.cross_val_score(self.model, data, label.astype(str), cv =4))

    def predict(self,ftest_,ltest_):
        #ltest=ltest_/ltest_.max(axis=0)
        #ftest=ftest_/ftest_.max(axis=0)
        ftest=ftest_
        ltest=ltest_
        predicted_data = []

        count=0
        for i in range(len(ftest)):
            predicted_data.append(self.model.predict(ftest.iloc[i].values.reshape(1,-1)))
        print predicted_data
        print ltest
        # for i in range(len(predcited_data)):
        #     if predcited_data[0][i]==ltest[0][i]:
        #         count+=1
        # print ("No of element Matching:",count)
        # print ("No of dissimilar elements:",len(predcited_data[0])-count)
        # print predcited_data
        # print ltest



def main():
    print ("Model Creation Start time:",datetime.datetime.now().time())
    testLSTM = speechLSTM()
    print ("Model Creation End time:",datetime.datetime.now().time())
    #dataframe = pd.DataFrame(testLSTM.load_data_file())
    #dataframe.to_csv("/media/vyassu/OS/Users/vyas/Documents/Assigments/BigData/Test-TrainingData_SVM.csv")
    dataframe = pd.read_csv("/media/vyassu/OS/Users/vyas/Documents/Assigments/BigData/Test-TrainingData_SVM.csv",usecols=['0','1','2'])
    print dataframe.groupby(['2']).count()

    ftest, ltest, ftrain, ltrain =  testLSTM.get_train_test_data(dataframe,0.20208335)
    print ("SVM Training start time:", datetime.datetime.now().time())
    testLSTM.trainNNet(ftrain,ltrain)
    print ("SVM Training end time:", datetime.datetime.now().time())
    testLSTM.predict(ftest,ltest)
    print ("Total processing end time:", datetime.datetime.now().time())
    #testLSTM.getIntermediateLayer()



if __name__ == '__main__':

    main()
