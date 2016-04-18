from pyspark.mllib.classification import SVMWithSGD, SVMModel,LogisticRegressionWithLBFGS,LinearClassificationModel
from pyspark.mllib.regression import LabeledPoint,LinearRegressionWithSGD,LassoWithSGD
import pyspark as py
import os
import hashlib
import glob as gb
import scipy.io.wavfile as sc
import SpeechPitchExtraction as lp
import analyse as an
import datetime
# Load and parse the data
def parsePoint(line):
    datapoints = line.split(",")
    try:
        for i in range(len(datapoints)):
            if i!=3:
                datapoints[i] = float(datapoints[i])
    except:
        j=0
    return LabeledPoint(float(int(hashlib.md5(datapoints[3]).hexdigest(), 16)/pow(10,38)), datapoints[1:3])

def getData(inputList):
    return LabeledPoint(inputList[2], inputList[0:2])

def dataconverter(filename):
    frate,inputdata = sc.read(filename=filename)
    pitch = lp.getPitch(filename, frate)
    emotion = ""
    loudness = abs(an.loudness(inputdata))
    filename = filename.split("/")[-1].split(".")[0]
    if filename[0] == "s":
        emotion = filename[0:2]
        emotion = ord(emotion[0])+ord(emotion[1])
    else:
        emotion = filename[0]
        emotion = float(ord(emotion))/100
    return [float(loudness), float(pitch), emotion]


working_directory = os.getcwd()
working_directory = working_directory+"/"


configuartion=py.SparkConf()                                # setting the Spark Configuration
sContext=py.SparkContext(conf=configuartion)                # setting the Spark context
sContext.defaultParallelism
print ("Data preprocessing start time:", datetime.datetime.now().time())
testdata = sContext.parallelize(gb.glob("/media/vyassu/OS/Users/vyas/Documents/Assigments/BigData/AudioData/DC/*")).map(dataconverter)
data = testdata.map(getData)
print ("Data preprocessing end time:", datetime.datetime.now().time())
print data.take(10)
# data1 = sContext.textFile(working_directory+"Test-TrainingData_SVM.csv")
#
#print testdata.count()
# #
# parsedData = data1.map(parsePoint)

# #print parsedData.take(100)
# print data.take(10)
# # Build the modelLogisticRegressionWithLBFGS
model = SVMWithSGD.train(data, iterations=15,validateData=True, intercept=True)
# #
# # Evaluating the model on training data
labelsAndPreds = data.map(lambda p: (p.label, model.predict(p.features)))
trainErr = labelsAndPreds.filter(lambda (v, p): v != p).count() / float(data.count())
print("Training Error = " + str(trainErr))
# #
# # # # Save and load model
# # # model.save(sc, "myModelPath")
# # # sameModel = SVMModel.load(sc, "myModelPath")