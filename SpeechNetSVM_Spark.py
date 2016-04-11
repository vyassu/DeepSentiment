from pyspark.mllib.classification import LogisticRegressionWithLBFGS, SVMModel
from pyspark.mllib.regression import LabeledPoint,LinearRegressionWithSGD
import pyspark as py
import os
import hashlib
import scipy.io.wavfile as sc

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

working_directory = os.getcwd()
working_directory = working_directory+"/"





configuartion=py.SparkConf()                                # setting the Spark Configuration
sContext=py.SparkContext(conf=configuartion)                # setting the Spark context
sContext.defaultParallelism
data = sContext.textFile(working_directory+"Test-TrainingData_SVM.csv")
testdata = sContext.textFile("/media/vyassu/OS/Users/vyas/Documents/Assigments/BigData/AudioData/KL/")

print testdata.take(1)

parsedData = data.map(parsePoint)
print parsedData.take(10)
# Build the modelLogisticRegressionWithLBFGS
model = LogisticRegressionWithLBFGS.train(parsedData, iterations=10,numClasses=7)

# Evaluating the model on training data
labelsAndPreds = parsedData.map(lambda p: (p.label, model.predict(p.features)))
trainErr = labelsAndPreds.filter(lambda (v, p): v != p).count() / float(parsedData.count())
print("Training Error = " + str(trainErr))

# # Save and load model
# model.save(sc, "myModelPath")
# sameModel = SVMModel.load(sc, "myModelPath")