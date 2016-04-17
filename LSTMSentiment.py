import cPickle,os
import preProcessor as pp
import numpy
import re
from keras.models import Sequential
from keras.layers.core import Dense, Activation,Dropout,TimeDistributedDense
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding


class LSTMSentiment:

    def __init__(self):
       self.in_dim = 500
       self.n_prev=25
       self.future=50
       out_dim = 1
       hidden_neurons = 500
       self.max_length = 100
       max_features = 20000
       
       # Initializing a sequential Model
       self.model = Sequential()
       self.model.add(Embedding(max_features, 128, input_length=self.max_length))
       self.model.add(Dropout(0.2))
       #self.model.add(LSTM(output_dim=128,input_dim=500,activation='relu'))
       self.model.add(LSTM(128))

       self.model.add(Dropout(0.2))
       self.model.add(Dense(1))
       self.model.add(Activation('linear'))


    def configureLSTMModel(self,TrainX,TrainY):
       print('Configuring the LSTM Model')
       self.model.compile(loss='binary_crossentropy', optimizer='adam',class_mode="binary")
       self.model.fit(TrainX, TrainY, nb_epoch=10,batch_size=32, show_accuracy=True,validation_split=0.3)
       #,validation_data =(ValidX,ValidY))


    def evaluateLSTMModel(self,TestX,TestY):
       obj_sc,acc = self.model.evaluate(TestX, TestY, batch_size=32,show_accuracy=True)
       print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
       print('Objective Score : ',obj_sc)
       print('Accuracy : ' ,acc)



    def predictSentiment(self,testX):
       sentiment = self.model.predict_classes(testX,batch_size=32)
       return sentiment

    def printSummary(self):
       print(self.model.summary())


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
   
def main():
   print('Initializing the LSTM Model')
   lstm = LSTMSentiment()
   
   print('Retrieving the Training and Test Data')
   path = os.getcwd()
   ((TrainX,TrainY),(TestX,TestY)) = lstm.getTrainTestData()
       

   print('#######################')
   print(TrainX)


   print('COnfiguring the LSTM model')
   lstm.configureLSTMModel(TrainX,TrainY)


   print('Evaluating the Model')
   lstm.evaluateLSTMModel(TestX,TestY)

   emotfile = open('/home/smeera380/spark-1.6.0/SpeechProject/SpeechSentimentAnalysis/aclImdb/emotion.txt',"rb");
   dataX = []
   dataY =[]
   dataX.append(emotfile.read())
   dataY.append('0') #Random output so as to call the pp.transformData function. This is not to be used anywhere
   (DataX,DataY) = pp.transformData(dataX,dataY)
   sentiment =lstm.predictSentiment(DataX)

   #lstm.printSummary()

   return sentiment
   
if __name__ =='__main__':
   main()
