import pyspark as py
import os
import glob as gb
import datetime
import re
import numpy as np
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
import cPickle
from sklearn.externals import joblib
from pyspark.mllib.classification import SVMWithSGD, SVMModel,LogisticRegressionWithLBFGS,LinearClassificationModel
from pyspark.mllib.regression import LabeledPoint,LinearRegressionWithSGD,LassoWithSGD
import py4j


def mapper(filename):
    f = open(filename, 'r')
    data = f.read()
    f.close()
    data.replace("<br />","")
    data = re.sub("[^\w]", " ",data)
    for element in data.split():
        yield(element,1)

def wordcount(inputline):
    for element in inputline.split():
        yield(inputline,1)
    return

def getdata(filename):
    f = open(filename, 'r')
    data = f.read()
    data.replace("<br />", "")
    data = re.sub("[^\w]", " ", data)
    f.close()
    return [data.split()]

def getIntDataFormat(data,dictionary):

    finalDataSet = []
    maxSentenceSize = 0
    for element in data:
        finalintList = []
        for word in element:
            if dictionary.get(word)!=None:
                finalintList.append(dictionary.get(word))
            else:
                finalintList.append(0)
            if len(finalintList)> maxSentenceSize:
                maxSentenceSize = len(finalintList)
        finalDataSet.append(finalintList)
    return finalDataSet,maxSentenceSize

def datapreprocessing(data,maxFeatures):
    print maxFeatures
    for i in range(len(data)):
        sentence = data[i]
        newData = []
        for j in range(maxFeatures-len(sentence)):
            sentence = sentence+[0]

        data[i] = sentence
    return data

def getLabel(maxFeatures, label,data):
    labelList=[]
    for i in range(maxFeatures):
        if label=="pos":
            labelList.append([data[i],1])
        else:
            labelList.append([data[i],0])
    return labelList

def getTestData(filename):
    f = open(filename, 'r')
    data = f.read()
    f.close()
    data.replace("<br />", "")
    data = re.sub("[^\w]", " ", data)
    return data.split()

def getLabelPoint(inputList):
    return LabeledPoint(inputList[1],inputList[0])

if __name__=="__main__":
    #model =
    py4j.java_gateway.launch_gateway
    finalDict={}
    details = {}
    if len(gb.glob("./diction*.pkl")) == 0:

        configuartion=py.SparkConf()                                # setting the Spark Configuration
        sContext=py.SparkContext(conf=configuartion)                # setting the Spark context
        sContext.defaultParallelism
        print ("Data preprocessing start time:", datetime.datetime.now().time())
        traindataPos = sContext.parallelize(gb.glob("/home/vyassu/Downloads/Telegram Desktop/aclImdb/train/pos/*.txt"))
        posData = traindataPos.flatMap(getdata)

        testdataPos = sContext.parallelize(gb.glob("/home/vyassu/Downloads/Telegram Desktop/aclImdb/test/pos/*.txt"))
        postestData = testdataPos.flatMap(getdata)

        newposData = traindataPos + testdataPos

        traindataNeg = sContext.parallelize(gb.glob("/home/vyassu/Downloads/Telegram Desktop/aclImdb/train/neg/*.txt"))
        negData = traindataNeg.flatMap(getdata)

        testdataNeg = sContext.parallelize(gb.glob("/home/vyassu/Downloads/Telegram Desktop/aclImdb/test/neg/*.txt"))
        negtestData = testdataNeg.flatMap(getdata)

        newNegData = traindataNeg + testdataNeg

        posDataFrequency = newposData.flatMap(mapper).reduceByKey(lambda a,b: a + b)
        negDataFrequency = newNegData.flatMap(mapper).reduceByKey(lambda a,b: a + b)

        dataFrequency = posDataFrequency + negDataFrequency
        dataFrequencySorted = dataFrequency.sortBy(lambda a: a[1],ascending=False)
        finalDict = {}

        newCount= 2

        for key,value in dataFrequencySorted.collect():
            finalDict.update({key:newCount})
            newCount+=1

        dictDump = open("dictionary.pkl", "wb")
        cPickle.dump(finalDict, dictDump, -1)

        finalposData,maxFeatures1 = getIntDataFormat(list(postestData.collect()),finalDict)
        finalnegData,maxFeatures2 = getIntDataFormat(list(negtestData.collect()),finalDict)
        finalposData,maxFeatures1 = getIntDataFormat(list(posData.collect()),finalDict)
        finalnegData,maxFeatures2 = getIntDataFormat(list(negData.collect()),finalDict)
        XtrainNeg,XtrainPos,XtestPos,XtestNeg=[],[],[],[]

        if maxFeatures1 < maxFeatures2:
            XtrainNeg = datapreprocessing(finalnegData,maxFeatures2)
            XtrainPos = datapreprocessing(finalposData,maxFeatures2)
            XtestNeg = datapreprocessing(finalnegData, maxFeatures2)
            XtestPos = datapreprocessing(finalposData, maxFeatures2)
            details.update({"maxfeature":maxFeatures2})
        elif maxFeatures1 > maxFeatures2:
            XtrainNeg = datapreprocessing(finalnegData, maxFeatures1)
            XtrainPos = datapreprocessing(finalposData, maxFeatures1)
            XtestNeg = datapreprocessing(finalnegData, maxFeatures1)
            XtestPos = datapreprocessing(finalposData, maxFeatures1)
            details.update({"maxfeature": maxFeatures1})
        else:
            XtrainNeg = datapreprocessing(finalnegData, maxFeatures1)
            XtrainPos = datapreprocessing(finalposData, maxFeatures1)
            XtestNeg = datapreprocessing(finalnegData, maxFeatures1)
            XtestPos = datapreprocessing(finalposData, maxFeatures1)
            details.update({"maxfeature": maxFeatures1})



        YtrainNeg,YtrainPos,YtestNeg,YtestPos = [],[],[],[]

        if len(XtrainPos)< len(XtrainNeg):
            print "Imbalance Dataset.. Balancing out commencing"
            XtrainNeg = XtrainNeg[0:len(XtrainPos)]
            XtrainNeg = getLabel(len(XtrainPos),"neg",XtrainNeg)
            XtrainPos = getLabel(len(XtrainPos),"pos",XtrainPos)
        elif len(XtrainPos)> len(XtrainNeg):
            print "Imbalance Dataset.. Balancing out commencing"
            XtrainPos = XtrainPos[0:len(XtrainNeg)]
            XtrainNeg = getLabel(len(XtrainNeg), "neg",XtrainNeg)
            XtrainPos = getLabel(len(XtrainPos), "pos",XtrainPos)
        else:
            print "Balance Dataset"
            XtrainNeg = getLabel(len(XtrainNeg), "neg",XtrainNeg)
            XtrainPos = getLabel(len(XtrainPos), "pos",XtrainPos)

        if len(XtestPos) < len(XtestNeg):
            print "Imbalance Dataset.. Balancing out commencing"
            XtestNeg = XtestNeg[0:len(XtestPos)]
            XtestNeg = getLabel(len(XtestPos), "neg",XtestNeg)
            XtestPos = getLabel(len(XtestPos), "pos",XtestPos)
        elif len(XtestPos) > len(XtestNeg):
            print "Imbalance Dataset.. Balancing out commencing"
            XtestPos = XtrainPos[0:len(XtestNeg)]
            XtestNeg = getLabel(len(XtestNeg), "neg",XtestNeg)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          
            XtestPos = getLabel(len(XtestNeg), "pos",XtestPos)
        else:
            print "Balance Dataset"
            XtestNeg = getLabel(len(XtestNeg), "neg",XtestNeg)
            XtestPos = getLabel(len(XtestNeg), "pos",XtestPos)


        train = XtrainPos+XtrainNeg
        test = XtestPos + XtestNeg
        train = sContext.parallelize(train)
        Xtrain = train.map(getLabelPoint)

        test = sContext.parallelize(test)
        Xtest = test.map(getLabelPoint)

        model = SVMWithSGD.train(Xtrain, iterations=1, validateData=True, intercept=True)
        #print model.predict(Xtest.map(lambda point: point.features)).collect()
        labelsAndPreds = Xtest.map(lambda p: (p.label, model.predict(p.features)))
        trainErr = labelsAndPreds.filter(lambda (v, p): v != p).count() / float(Xtest.count())
        print("Training Error = " + str(trainErr))
    # else:
    #     detailsFile = open("./datadetails.pkl", 'rb')
    #     dictFile = open("./dictionary.pkl", 'rb')
    #     model = joblib.load("./Models/SVM_SpeechText_Model.pkl")
    #     details = cPickle.load(detailsFile)
    #     finalDict = cPickle.load(dictFile)
    ########## End of If Loop MOdel TRAINED ##############################
    # dataList=[]
    # testDataList = getTestData("./Data/2_7.txt")
    # dataList.append(testDataList)
    # finalData, maxFeatures = getIntDataFormat(dataList, finalDict)
    #
    # modelTrainFeatures = details.get("maxfeature")
    # if modelTrainFeatures > maxFeatures:
    #     Xtest = np.array(datapreprocessing(finalData, modelTrainFeatures))
    # else:
    #     Xtest = np.array(finalData[0:modelTrainFeatures])
    # print Xtest
    # print model.score(Xtest,[1])
