import cPickle,os
import numpy as np
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
from sklearn.externals import joblib
import logging as l

class SVMSentiment:

    def __init__(self):
       l.getLogger("SVMSentimentAnalysis")
       l.basicConfig(level=l.ERROR)
       l.debug('Initializing the SVM Model')
       self.max_length = 500
       self.batch_size=50
       self.model = OneVsRestClassifier(svm.SVC(kernel='rbf',gamma=3,C = 0.5,tol=0.0001,cache_size=5000))

    def configureSVMModel(self,TrainX,TrainY,validX,validY):
       l.debug('Configuring the SVM Model')
       currPath = os.getcwd()
       currFiles =  os.listdir(currPath)
       if(currFiles.count('SVMScores.pkl')==0):
          self.model.fit(TrainX, TrainY)
          # Saving model scores
          joblib.dump(self.model,currPath+'/SpeechTextModels/SVMScores.pkl')
       else:
          l.debug('Loading already existing Model')
          self.model = joblib.load(currPath+'/SpeechTextModels//SVMScores.pkl')
       

    def evaluateSVMModel(self,TestX,TestY):
       l.debug("Model Score:::%s",self.model.score(TestX, TestY))
       predicted_data=[]
       for i in range(len(TestX)):
          predicted_data.append(list([self.model.predict (TestX[i].reshape(1,-1)) ,TestY[i]]) )
       l.debug("Current Model Prediction::: %s",str(predicted_data))

    def predictSentiment(self,dataX):
       for i in range(len(dataX)):
         predicted_data = self.model.predict(dataX[i].reshape(1,-1))
       return predicted_data

    def getTrainTestData(self):
       l.debug('Loading Training and Test data')
       (trainX,trainY) = cPickle.load(open('trainingdata.pkl','rb'))
       (testX,testY)  = cPickle.load(open('testingdata.pkl','rb'))
       return ((trainX,trainY),(testX,testY))

    def getValidationData(self,dataX,dataY):
       return dataX[0:self.batch_size,:],dataY[0:self.batch_size,:]

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

def datapreprocessing(data, maxFeatures):
   for i in range(len(data)):
          sentence = data[i]
          for j in range(maxFeatures - len(sentence)):
             sentence = sentence + [0]
          data[i] = sentence
   return data

# arg rpresents the path of the filename which has the text converted from the speech signal.
def main(inputFileName,training,log_level,*args):
   l.basicConfig(level=log_level)
   cwd = os.getcwd()
   svm = SVMSentiment()
   l.info('Retrieving the Training-Test Data for Speech2Text Model')
   ((TrainX,TrainY),(TestX,TestY)) = svm.getTrainTestData()

   l.info('Retrieving Validation Data for Speech2Text Model')
   validX, validY = svm.getValidationData(TrainX,TrainY)

   l.info('Configuring the SVM Model')
   svm.configureSVMModel(TrainX,TrainY,validX,validY)
   svm.evaluateSVMModel(TestX,TestY)

   if inputFileName=='':
       l.error("Input data not specified")
       raise RuntimeError ("InputFile not specified")
       return
   else:
      emotfile = open(inputFileName,"rb");
      dataList = emotfile.read().split()
      l.info(" Finished Loading the saved Speech2Text Model.")
      finalDict = cPickle.load(open(cwd+'/dictionary.pkl','rb'))
      detailsFile = open("./datadetails.pkl", 'rb')
      details = cPickle.load(detailsFile)
      finalData, maxFeatures = getIntDataFormat(dataList, finalDict)

      modelTrainFeatures = details.get("maxfeature")
      if modelTrainFeatures > maxFeatures:
         Xtest = np.array(datapreprocessing(finalData, 100))
      else:
         Xtest = np.array(finalData[0:modelTrainFeatures])
      return svm.predictSentiment(Xtest)
   
if __name__ =='__main__':
   arg = "/home/vyassu/PycharmProjects/DeepSentiment/Data/113550507951.txt"
   print(main(arg))
