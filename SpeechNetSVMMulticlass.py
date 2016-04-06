import numpy as np
import pandas as pd
import datetime
import analyse as an
import glob as gb
import scipy.io.wavfile as sc
import SpeechPitchExtraction as lp
import random
import pickle
from sklearn import svm
from sklearn import cross_validation
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
from sklearn.preprocessing import MultiLabelBinarizer

class speechLSTM:
    # Initializing the LSTM Model
    def __init__(self):
       self.prevData = 100
       self.batchsize=200
       self.model = OneVsRestClassifier(svm.SVC(kernel='rbf',gamma=1,C = 1,tol=0.0001,cache_size=5000)  )     #self.model = OneVsRestClassifier(LinearSVC(random_state=0))
       self.working_directory = "/media/vyassu/OS/Users/vyas/Documents/Assigments/BigData/"
       self.model_prediction_score = {}

    def get_Model_Score(self):
        filename = self.working_directory + "Models/scorefile.txt"
        return pickle.load(open(filename, "rb"))

    def set_Model_Score(self):
        filename = self.working_directory+"Models/scorefile.txt"
        pickle.dump(self.model_prediction_score, open(filename, "wb"))

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

    def trainNNet(self,data,label,feature_name):
        filenamelist =  gb.glob(self.working_directory+"Models/*")
        filename = "Models/SVM_" + feature_name + ".pkl"
        print filenamelist.count(self.working_directory+"Models/SVM_"+feature_name+".pkl")
        if filenamelist.count(self.working_directory+"Models/SVM_"+feature_name+".pkl") == 0:
            X_train, X_test, y_train, y_test = cross_validation.train_test_split(data, label.astype(str), test_size = 0.045, random_state = 0)
            self.model.fit(X_train,y_train)
            print("score",self.model.score(X_test,y_test))
            print (cross_validation.cross_val_score(self.model, data, label.astype(str), cv =4))
            joblib.dump(self.model,  self.working_directory+filename)
        else:
            self.model = joblib.load(self.working_directory+filename)
            print "model already exists for feature "+feature_name+" !! training exiting"

    def predict(self,ftest,ltest,data,feature_name):
        predicted_data = []

        for i in range(len(ftest)):
            predicted_data.append(self.model.predict(ftest.iloc[i].values.reshape(1,-1)))
        print predicted_data
        print self.model.predict(data.iloc[0].values.reshape(1,-1))
        score = self.model.score(ftest, ltest)
        print("score", score)
        self.model_prediction_score.update({feature_name:score})

    def predict_emotion(self,data):
        emotion_list=[]
        for modelfilepath in gb.glob(self.working_directory+"Models/*.pkl"):
            print modelfilepath
            emotion = modelfilepath.split("/")[-1].split(".")[0]
            model = joblib.load(modelfilepath)
            modelprediction = model.predict(data.values.reshape(1,-1))
            print modelprediction
            if modelprediction[0] !='NA':
                emotion_list.append(modelprediction[0])
            print emotion_list
        return emotion_list



def main():
    attributes =['a','d','h','su','sa','f']
    emotions_mapping = {"a":"Angry","d":"Disgust","h":"happy","su":"surprise","sa":"sadness","f":"fear"}
    working_directory = "/media/vyassu/OS/Users/vyas/Documents/Assigments/BigData/"
    print ("Model Creation Start time:",datetime.datetime.now().time())
    testLSTM = speechLSTM()
    print ("Model Creation End time:",datetime.datetime.now().time())
    #dataframe = pd.DataFrame(testLSTM.load_data_file())
    #dataframe.to_csv("/media/vyassu/OS/Users/vyas/Documents/Assigments/BigData/Test-TrainingData_SVM.csv")
    dataframe = pd.read_csv("/media/vyassu/OS/Users/vyas/Documents/Assigments/BigData/Test-TrainingData_SVM.csv",usecols=['0','1','2'])
    print dataframe.groupby(['2']).count()
    modelList =  gb.glob(working_directory+"Models/*.pkl")
    if len(modelList)==0:
        for feature in attributes:
            df= dataframe.groupby(['2']).get_group(feature)
            df1 = dataframe.groupby(['2']).get_group('n')
            df1 = pd.concat([df, df1],ignore_index=False)
            df1 = df1.replace('n',"NA")
            print ("File Data load End time:",datetime.datetime.now().time())
            ftest, ltest, ftrain, ltrain =  testLSTM.get_train_test_data(df1,0.05)
            print ("Test and Train data created:", datetime.datetime.now().time())
            testLSTM.trainNNet(ftrain,ltrain,feature_name=feature)
            testLSTM.predict(ftest,ltest,ftrain,feature_name=feature)
            testLSTM.set_Model_Score()

    rand =  random.randint(0,len(dataframe))
    print dataframe.iloc[rand:rand+1, 0:2]
    print dataframe.iloc[rand:rand+1,2]
    emotionData = {}
    emotionList = testLSTM.predict_emotion(dataframe.iloc[rand:rand+1, 0:2])
    emtionscore = testLSTM.get_Model_Score()
    if len(emotionList)==0:
        emotionData = {"Neutral":"1.00"}
    else:
        for emotions in emotionList:
            emo = emotions_mapping.get(emotions)
            emoscore = emtionscore.get(emotions)
            emotionData.update({emo:emoscore})

    print emotionData




if __name__ == '__main__':

    main()
