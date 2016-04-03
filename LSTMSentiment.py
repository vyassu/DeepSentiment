import cPickle,os
import Preprocessor as pp
import numpy
from keras.models import Sequential
from keras.layers.core import Dense, Activation,Dropout,TimeDistributedDense
from keras.layers.recurrent import LSTM



class LSTMSentiment:

    def __init__(self):
       self.in_dim = 1
       self.n_prev=25
       self.future=50
       out_dim = 1
       hidden_neurons = 300
       
       # Initializing a sequential Model
       self.model = Sequential()
       self.model.add(Dense(output_dim=1,input_dim=2,activation='relu'))
       self.model.add(Activation("relu"))
       self.model.add(Dense(output_dim=10))
       self.model.add(Activation("softmax"))
       self.model.add(Dropout(0.5))


    def configureLSTMModel(self,trainX,trainY):
       self.model.compile(loss='categorical_crossentropy', optimizer='sgd')
       self.model.fit(trainX, trainY, nb_epoch=5, batch_size=32)


    def evaluateLSTMModel(self,testX,testY):
       objective_score = self.model.evaluate(X_test, Y_test, batch_size=32)
       print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
       print(objective_score)


    def predictSentiment(self,testX):
       sentiment = self.model.predict_classes(testX,batch_size=32)
       print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
       print(sentiment)


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

       #print('############# Counts and Keys ################')
       #print(counts ,'::::::::: ',keys)

       sorted_idx = numpy.argsort(counts)[::-1]
       #print('################ SortedId ###################')
       #print(sorted_idx)

       worddict = dict()


       for idx, ss in enumerate(sorted_idx):
          worddict[keys[ss]] = idx+2  # leave 0 and 1 (UNK)
       print('###################### Dictionary ######################')
       print(worddict)

       print numpy.sum(counts), ' total words ', len(keys), ' unique words'

       return worddict

    # Transforms the sentences into number vectors where the number represents the value corresponding to the word in the Dictionary built above
    def transformData(self,dataX,dataY,worddict):
       transformedDataX = [None] * len(dataX)
       transformedDataY = dataY
       for ind,sen in enumerate(dataX):
          words = sen.lower().split()
          transformedDataX[ind]=[]
          for w in words:
             if w in worddict:
                transformedDataX[ind].append(worddict[w])
             else:
                transformedDataX[ind].append(1)
          
          #transformedData[ind] = [worddict[w] if w in worddict else 1 for w in words]
       #print('############################# Transformed Data ###################')
       #print(transformedData)
       return (transformedDataX, transformedDataY)



    def prepareData(self,dataX):
       wordList=[]
       for i in xrange(0,len(dataX)):
          wordList[i] = re.sub("[^\w]", " ",  dataX[i]).split()
          print(dataX[i],' :::::::::::::::::::::   ',wordList[i])       


def main():
   print('Initializing the LSTM Model')
   lstm = LSTMSentiment()
   
   print('Retrieving the Training and Test Data')
   path = os.getcwd()
   ((trainX,trainY),(testX,testY)) = lstm.getTrainTestData()
       
   worddict = dict()
   worddict = lstm.build_dict(trainX,testX)
   (TrainX,TrainY) = lstm.transformData(trainX,trainY,worddict)

   print('********************** Training Data *********************')
   for i in xrange(0,len(TrainX)):
       print(TrainX[i] , '  :  :  ' , TrainY[i])

   (TestX,TestY) = lstm.transformData(testX,testY,worddict)
     
   print('********************** Testing Data **********************')
   for i in xrange(0,len(TestX)):
       print(TestX[i] , '  :  :  ' , TestY[i])   

   
   
if __name__ =='__main__':
   main()
