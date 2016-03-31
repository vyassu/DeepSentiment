import cPickle,os
import Preprocessor as pp
from keras.models import Sequential
from keras.layers.core import Dense, Activation,Dropout,TimeDistributedDense
from keras.layers.recurrent import LSTM

class LSTMSentiment:

    def __init__(self):
       self.in_dim = 19
       self.n_prev=25
       self.future=50
       out_dim = 1
       hidden_neurons = 300
       '''
       self.model = Sequential()
       self.model.add(LSTM(output_dim=hidden_neurons,input_dim = self.in_dim,return_sequences=False))
       self.model.add(Dropout(0.5))
       self.model.add(Dense(output_dim=out_dim,input_dim=300,activation='linear'))
       self.model.compile(loss="categorical_crossentropy", optimizer='adam',class_mode="binary")
       #self.model.compile(optimizer="sgd")
       '''


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
   ((trainX,trainY),(testX,testY)) = lstm.getTrainTestData()
       
   print('******************************* Training Data ***************************')
   for i in xrange(0,len(trainX)):
     print(trainX[i] ,':::::::::' ,trainY[i])

   print('******************************* Test Data ***************************')
   for i in xrange(0,len(testX)):
     print(testX[i] ,':::::::::' ,testY[i])
   
if __name__ =='__main__':
   main()
