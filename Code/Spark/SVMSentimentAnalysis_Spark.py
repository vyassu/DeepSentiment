import findspark
findspark.init()
import glob as gb
import datetime,re,cPickle
import numpy as np
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
from sklearn.externals import joblib
import pyspark as py
import logging

logging.getLogger("SVMSpeech2TextSpark")
logging.basicConfig(level=logging.DEBUG)

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
    logging.debug("Starting Text to Integer Conversion")
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
    logging.debug("Max Features::%s",maxFeatures)
    for i in range(len(data)):
        sentence = data[i]
        for j in range(maxFeatures-len(sentence)):
            sentence = sentence+[0]
        data[i] = sentence
    return data

def getLabel(maxFeatures, label):
    labelList=[]
    for i in range(maxFeatures):
        if label=="pos":
            labelList.append(1)
        else:
            labelList.append(0)
    return labelList

def getTestData(filename):
    f = open(filename, 'r')
    data = f.read()
    f.close()
    data.replace("<br />", "")
    data = re.sub("[^\w]", " ", data)
    return data.split()

def main(inputFileName,training,log_level,*args):
    logging.basicConfig(level=log_level)
    details = {}
    if training == True:
        configuartion=py.SparkConf()                                # setting the Spark Configuration
        sContext=py.SparkContext(conf=configuartion)                # setting the Spark context
        sContext.defaultParallelism
        logging.debug ("Data preprocessing start time:", datetime.datetime.now().time())
        traindataPos = sContext.parallelize(gb.glob(args[0]+"*.txt"))
        posData = traindataPos.flatMap(getdata)

        testdataPos = sContext.parallelize(gb.glob(args[1]+"*.txt"))
        postestData = testdataPos.flatMap(getdata)

        newposData = traindataPos + testdataPos

        traindataNeg = sContext.parallelize(gb.glob(args[2]+"*.txt"))
        negData = traindataNeg.flatMap(getdata)

        testdataNeg = sContext.parallelize(gb.glob(args[3]+"*.txt"))
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
            logging.debug("Imbalance Dataset.. Balancing out commencing")
            XtrainNeg = XtrainNeg[0:len(XtrainPos)]
            YtrainNeg = getLabel(len(XtrainPos),"neg")
            YtrainPos = getLabel(len(XtrainPos),"pos")
        elif len(XtrainPos)> len(XtrainNeg):
            logging.debug("Imbalance Dataset.. Balancing out commencing")
            XtrainPos = XtrainPos[0:len(XtrainNeg)]
            YtrainNeg = getLabel(len(XtrainNeg), "neg")
            YtrainPos = getLabel(len(XtrainNeg), "pos")
        else:
            logging.debug( "Balance Dataset")
            YtrainNeg = getLabel(len(XtrainNeg), "neg")
            YtrainPos = getLabel(len(XtrainNeg), "pos")

        if len(XtestPos) < len(XtestNeg):
                logging.debug( "Imbalance Dataset.. Balancing out commencing")
                XtestNeg = XtestNeg[0:len(XtestPos)]
                YtestNeg = getLabel(len(XtestPos), "neg")
                YtestPos = getLabel(len(XtestPos), "pos")
        elif len(XtestPos) > len(XtestNeg):
                logging.debug( "Imbalance Dataset.. Balancing out commencing")
                XtestPos = XtrainPos[0:len(XtestNeg)]
                YtestNeg = getLabel(len(XtestNeg), "neg")
                YtestPos = getLabel(len(XtestNeg), "pos")
        else:
            logging.debug( "Balance Dataset")
            YtestNeg = getLabel(len(XtestNeg), "neg")
            YtestPos = getLabel(len(XtestNeg), "pos")


        Xtrain = XtrainPos+XtrainNeg
        Ytrain = YtrainPos+YtrainNeg
        Xtest = XtestPos + XtestNeg
        Ytest = YtestPos + YtestNeg
        Xtrain = np.array(Xtrain)
        Ytrain = np.array(Ytrain)

        model = OneVsRestClassifier(svm.SVC(kernel='rbf',gamma=3,C = 0.5,tol=0.0001,cache_size=5000)  )
        model.fit(Xtrain,Ytrain)
        logging.info("Speech2Text Model Training Completed")
        logging.info("Speech2Text Model Score::"+model.score(Xtest, Ytest))

        details.update({"score":model.score(Xtest, Ytest)})

        dictDump = open("datadetails.pkl", "wb")
        cPickle.dump(details, dictDump, -1)

        logging.info("Saved the Speech2Text Model State")
        joblib.dump(model, "./SpeechTextModels/SVM_SpeechText_Model.pkl")
    else:
        detailsFile = open("./datadetails.pkl", 'rb')
        dictFile = open("./dictionary.pkl", 'rb')
        model = joblib.load("./SpeechTextModels/SVM_SpeechText_Model.pkl")
        logging.info("Finished loading Speech2Text Model State")
        details = cPickle.load(detailsFile)
        finalDict = cPickle.load(dictFile)

    ########## End of If Loop MOdel TRAINED ##############################
    dataList=[]
    dataList.append(getTestData(inputFileName))
    finalData, maxFeatures = getIntDataFormat(dataList, finalDict)

    modelTrainFeatures = details.get("maxfeature")
    if modelTrainFeatures > maxFeatures:
        Xtest = np.array(datapreprocessing(finalData, modelTrainFeatures))
    else:
        Xtest = np.array(finalData[0:modelTrainFeatures])
    return model.predict(Xtest)

if __name__ == '__main__':
   print main("/home/vyassu/PycharmProjects/DeepSentiment/Data/113550507951.txt",False,10,"","","","")