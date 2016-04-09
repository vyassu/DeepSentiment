import cPickle,os
import preProcessor as pp
#from Preprocessor import transformData
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
       print('################### Test #####################')
       print(currFiles.count('SVMScores.pkl'))
       if(currFiles.count('SVMScores.pkl')==0):
          self.model.fit(TrainX, TrainY)
          # Saving model scores
          joblib.dump(self.model,currPath+'/SVMScores.pkl')
       else:
          print('Loading already existing Model')
          self.model = joblib.load(currPath+'/SVMScores.pkl')
       

    def evaluateSVMModel(self,TestX,TestY):
       print self.model.score(TestX, TestY)

       predicted_data=[]
       for i in range(len(TestX)):
          predicted_data.append(list([self.model.predict (TestX[i].reshape(1,-1)) ,TestY[i]]) )

       print "Predicted Data"
       print predicted_data
       #print TestY

    def predictSentiment(self,dataX,dataY):
       print('@@@@@@@@@@@@@@@@ Length of test data : ',len(dataX))
       for i in range(len(dataX)):
         predicted_data = self.model.predict(dataX[i].reshape(1,-1))
         expected_out = dataY[i]

       print('############### Predicted data :',predicted_data,' ; ; ',expected_out)
       return predicted_data

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




    def getValidationData(self,dataX,dataY):
       return dataX[0:self.batch_size,:],dataY[0:self.batch_size,:]

# arg rpresents the path of the filename which has the text converted from the speech signal.
def main(arg):
   print('Initializing the LSTM Model')
   cwd = os.getcwd()
   svm = SVMSentiment()
   
   print('Retrieving the Training and Test Data')
   path = os.getcwd()
   ((TrainX,TrainY),(TestX,TestY)) = svm.getTrainTestData()

   print('Getting the Validation Data')
   validX, validY = svm.getValidationData(TrainX,TrainY)

   print('Configuring the SVM Model')
   svm.configureSVMModel(TrainX,TrainY,validX,validY)

   #print('Evaluating the Model')
   svm.evaluateSVMModel(TestX,TestY)

   if arg=='':
      return
   else:
      emotfile = open(arg,"rb");
      dataX = []
      dataY =[]
      dataX.append(emotfile.read())
      dataY.append('0') #Random output so as to call the pp.transformData function. This is not to be used anywhere
      worddict = cPickle.load(open(cwd+'/dictionary.pkl','rb'))
      (DataX,DataY) = pp.transformData(dataX,dataY,worddict)       
      prediction = svm.predictSentiment(DataX,DataY)
      return prediction


   
if __name__ =='__main__':
   arg ='/home/smeera380/spark-1.6.0/SpeechProject/SpeechSentimentAnalysis/aclImdb/emotion.txt'
   main(arg)
