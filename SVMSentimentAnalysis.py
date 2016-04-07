import cPickle,os
import Preprocessor as pp
import numpy
import re
from keras.models import Sequential
from keras.layers.core import Dense, Activation,Dropout,TimeDistributedDense
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding
from sklearn import svm
from sklearn import cross_validation
from sklearn.multiclass import OneVsRestClassifier
from sklearn.externals import joblib


class SVMSentiment:

    def __init__(self):
       self.max_length = 500
       self.batch_size=50
       self.model = OneVsRestClassifier(svm.SVC(kernel='rbf',gamma=1,C = 1,tol=0.0001,cache_size=5000)  )


    def configureSVMModel(self,TrainX,TrainY,validX,validY):
       print('Configuring the SVM Model')
       currPath = os.getcwd()
       currFiles =  os.listdir(currPath)
       if(currFiles.count('SVMScores.pkl')==0):
          self.model.fit(TrainX, TrainY)
          # Saving model scores
          joblib.dump(self.model,'SVMScores.pkl')
       else:
          print('Loading already existing Model')
          self.model = joblib.load(currPath+'/SVMScores.pkl')
       

    def evaluateSVMModel(self,TestX,TestY):
       print self.model.score(TestX, TestY)

       predicted_data=[]
       for i in range(len(TestX)):
          predicted_data.append(list([self.model.predict(TestX[i].reshape(1,-1)),TestY[i]]))

       print "Predicted Data"
       print predicted_data
       #print TestY


    def getTrainTestData(self):
       print('Loading Training and Test data')
       trainX=[]
       trainY=[]
       testX=[]
       testY = []

       f= open('trainingdata.pkl','rb')
       (trainX,trainY) = cPickle.load(f)
       
       f= open('testingdata.pkl','rb')
       (testX,testY)  = cPickle.load(f)

       return ((trainX,trainY),(testX,testY))


    def build_dict(self,trainX,testX):
       sentences =[]
       sentences = trainX + testX
       wordCnt = dict()
       # Splitting each sentences into words and getting the word count.
       for i in sentences:
         words = i.lower().split()
         for w in words:
            if w not in wordCnt:
               wordCnt[w] = 1
            else:
               wordCnt[w] +=1

       counts = wordCnt.values()
       keys = wordCnt.keys()
       worddict = dict()
       sorted_idx = numpy.argsort(counts)[::-1]

       for idx, ss in enumerate(sorted_idx):
          worddict[keys[ss]] = idx+2

       return worddict

    # Transforms sentences into Integer vectors where number represents value corresponding to the word in the Dictionary built above
    def transformData(self,dataX,dataY,worddict):
       transformedDataX = [None] * len(dataX)
       transformedDataY = dataY
       for ind,sen in enumerate(dataX):
          words = re.sub("[^\w]", " ", sen).lower().split()
          transformedDataX[ind]=[]
          for w in words:
             if w in worddict:
                transformedDataX[ind].append(worddict[w])
             else:
                transformedDataX[ind].append(1)
          
       #Converting the length of the transformed data to maximum length
       transX = []
       for i in transformedDataX:
          transLen = len(i)
          if(transLen < self.max_length): #Pad zeroes to the data vector
              transX.append([0]*(self.max_length - transLen) + i)
          elif transLen > self.max_length:
              j = i
              del j[self.max_length:]
              transX.append(j)
          #transformedData[ind] = [worddict[w] if w in worddict else 1 for w in words]
   
       transX = numpy.array(transX)
       transformedDataY = numpy.array(transformedDataY)
       transformedDataY = transformedDataY.reshape(transformedDataY.shape[0],1)

       return (transX, transformedDataY)


    def prepareData(self,dataX):
       wordList=[]
       for i in xrange(0,len(dataX)):
          wordList[i] = re.sub("[^\w]", " ",  dataX[i]).split()


    def getValidationData(self,dataX,dataY):
       return dataX[0:self.batch_size,:],dataY[0:self.batch_size,:]


def main():
   print('Initializing the LSTM Model')
   svm = SVMSentiment()
   
   print('Retrieving the Training and Test Data')
   path = os.getcwd()
   ((trainX,trainY),(testX,testY)) = svm.getTrainTestData()

   worddict = dict()
   worddict = svm.build_dict(trainX,testX)

   print('Transforming Training and Test Data')
   (TrainX,TrainY) = svm.transformData(trainX,trainY,worddict)
   (TestX,TestY) = svm.transformData(testX,testY,worddict)

   print('Getting the Validation Data')
   validX, validY = svm.getValidationData(TrainX,TrainY)

   print('Configuring the SVM Model')
   svm.configureSVMModel(TrainX,TrainY,validX,validY)

   print('Evaluating the Model')
   svm.evaluateSVMModel(TestX,TestY)

   
if __name__ =='__main__':
   main()
