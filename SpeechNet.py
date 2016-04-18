from keras.models import Sequential
from keras.layers.core import Dense, Activation,Dropout ,TimeDistributedDense,Reshape,RepeatVector,TimeDistributedMerge
from keras.layers.recurrent import LSTM,GRU
from keras.layers import Embedding
from random import random
import numpy as np
import pandas as pd
import datetime
import analyse as an
import glob as gb
import scipy.io.wavfile as sc
import hashlib
import SpeechPitchExtraction as lp
from keras import backend as K
import random


class speechLSTM:
    # Initializing the LSTM Model
    def __init__(self):
       self.prevData = 30
       self.batchsize=200
       self.model = Sequential()

    def build_nnet(self):
       self.model.add(LSTM(300,return_sequences=True, stateful=True,
                      batch_input_shape=(self.batchsize, self.prevData, 2)))
       self.model.add(Activation("linear"))
       self.model.add(Dropout(0.5))
       # self.model.add(LSTM(400,return_sequences=True,stateful=True))
       # self.model.add(Activation("linear"))
       # self.model.add(Dropout(0.5))
       # self.model.add(LSTM(500, return_sequences=True, stateful=True))
       # self.model.add(Activation("linear"))
       # self.model.add(Dropout(0.5))
       self.model.add(LSTM(400, return_sequences=True, stateful=True))
       self.model.add(Activation("relu"))
       # self.model.add(Dropout(0.5))
       # self.model.add(LSTM(700, return_sequences=True, stateful=True))
       # self.model.add(Activation("linear"))
       self.model.add(LSTM(500, return_sequences=False, stateful=True))
       self.model.add(Activation("linear"))
       self.model.add(Dropout(0.5))
       self.model.add(Dense(1, activation='sigmoid'))
       self.model.compile(loss='binary_crossentropy', optimizer='adadelta')


    def load_data_file(self):
        outputdata = []                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         
        for f in gb.glob("/media/vyassu/OS/Users/vyas/Documents/Assigments/BigData/AudioData/KL/*.wav"):
            frate, inputdata = sc.read(f)
            pitch=lp.getPitch(f,frate)
            emotion = ""
            loudness = abs(an.loudness(inputdata))
            filename = f.split("/")[-1].split(".")[0]
            if filename[0] == "s":
                emotion = filename[0:2]
                emotion = float(int(hashlib.md5(emotion).hexdigest(), 16))
            else:
                emotion = filename[0]
                emotion =  float(int(hashlib.md5(emotion).hexdigest(), 16))
            outputdata.append(list([loudness,pitch, emotion]))

        return outputdata

    def get_train_test_data(self,data,percent_split):
        ftestList,ltestlist,fvalidList,lvalidList,ftrainList,ltrainList=[],[],[],[],[],[]
        noOfTrainSamples = len(data)*(1-percent_split)

        noOfTestSamples = len(data)-noOfTrainSamples
        self.batchsize = int(noOfTestSamples)

        noOfTrainSamples = int((noOfTrainSamples - self.prevData)/noOfTestSamples)

        for i in range(int(noOfTrainSamples)*self.batchsize):
            #ltrainList.append(data.iloc[i:i+self.prevData, 2:].as_matrix())
            ftrainList.append(data.iloc[i:i+self.prevData, 0:2].as_matrix())
        ltrainList = data.iloc[0:int(noOfTrainSamples)*self.batchsize, 2:].values

        for i in range(self.batchsize):
            fvalidList.append(data.iloc[i:i + self.prevData, 0:2].as_matrix())
        lvalidList = data.iloc[0:self.batchsize, 2:].values

        randNum = random.randint(0,noOfTrainSamples)

        for i in range(randNum,randNum+self.batchsize):
            ftestList.append(data.iloc[i:i + self.prevData, 0:2].as_matrix())
        ltestlist = data.iloc[randNum: randNum+self.batchsize, 2:].values

        return np.array(ftestList),np.array(ltestlist),np.array(ftrainList),np.array(ltrainList),np.array(fvalidList),np.array(lvalidList)

    def trainNNet(self,data_,label_,valid_data,valid_label):
        data = data_/data_.max(axis=0)
        label = label_/label_.max(axis=0)
        valid_data = valid_data/valid_data.max(axis=0)
        valid_label = valid_label/valid_label.max(axis=0)
        self.model.fit(data, label, batch_size=self.batchsize, nb_epoch=5,validation_data=(valid_data,valid_label),show_accuracy=True,shuffle=False)

    def predict(self,ftest_,ltest_):
        ltest=ltest_/ltest_.max(axis=0)
        ftest=ftest_/ftest_.max(axis=0)
        count=0
        predcited_data= self.model.predict_on_batch(ftest)
        print ("Score:",self.model.evaluate(ftest,ltest, show_accuracy=True))
        for i in range(len(predcited_data)):
            if predcited_data[0][i]==ltest[0][i]:
                count+=1
        print ("No of element Matching:",count)
        print ("No of dissimilar elements:",len(predcited_data[0])-count)
        print predcited_data
        print ltest

    def saveModel(self):
        self.model.save_weights("/media/vyassu/OS/Users/vyas/Documents/Assigments/BigData/", overwrite=False)

    def getIntermediateLayer(self):
        get_3rd_layer_output = K.function([self.model.layers[0].input],
                                          [self.model.layers[3].get_output(train=False)])
        #layer_output = get_3rd_layer_output[0]
        print K.get_value(get_3rd_layer_output)


def main():
    print ("Model Creation Start time:",datetime.datetime.now().time())
    testLSTM = speechLSTM()
    print ("Model Creation End time:",datetime.datetime.now().time())
    #dataframe = pd.DataFrame(testLSTM.load_data_file())
    #dataframe.to_csv("/media/vyassu/OS/Users/vyas/Documents/Assigments/BigData/Test-TrainingData.csv")
    dataframe = pd.read_csv("/media/vyassu/OS/Users/vyas/Documents/Assigments/BigData/Test-TrainingData.csv",usecols=['0','1','2'])
    print ("File Data load End time:",datetime.datetime.now().time())
    ftest, ltest, ftrain, ltrain,fvalid,lvalid =  testLSTM.get_train_test_data(dataframe,0.2)
    print ("Test and Train data created:",datetime.datetime.now().time())
    print ("LSTM Model creation started:", datetime.datetime.now().time())
    testLSTM.build_nnet()
    print ("LSTM Model creation ended:", datetime.datetime.now().time())
    print ("LSTM Model Training started:", datetime.datetime.now().time())
    testLSTM.trainNNet(ftrain,ltrain,fvalid,lvalid)
    print ("LSTM Model Training ended:", datetime.datetime.now().time())

    testLSTM.predict(ftest,ltest)
    #testLSTM.getIntermediateLayer()



if __name__ == '__main__':

    main()
