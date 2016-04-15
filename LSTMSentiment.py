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
       self.model.add(LSTM(300))

       self.model.add(Dropout(0.2))
       self.model.add(Dense(1))
       self.model.add(Activation('linear'))


    def configureLSTMModel(self,TrainX,TrainY):
       print('Configuring the LSTM Model')
       self.model.compile(loss='binary_crossentropy', optimizer='adam') #class_mode ="binary"
       #,class_mode ="binary")
       self.model.fit(TrainX, TrainY, nb_epoch=15,batch_size=32, show_accuracy=True,validation_split=0.2)
       #,validation_data =(ValidX,ValidY))


    def evaluateLSTMModel(self,TestX,TestY):
       obj_sc,acc = self.model.evaluate(TestX, TestY, batch_size=32,show_accuracy=True)
       print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
       print('Objective Score : ',obj_sc)
       print('Accuracy : ' ,acc)



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
    '''
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
          worddict[keys[ss]] = idx+2

       print numpy.sum(counts), ' total words ', len(keys), ' unique words'

       return worddict

    # Transforms sentences into number vectors where number represents value corresponding to the word in the Dictionary built above
    def transformData(self,dataX,dataY,worddict):
       transformedDataX = [None] * len(dataX)
       transformedDataY = dataY
       for ind,sen in enumerate(dataX):
          #words = sen.lower().split()
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
              #print('@@@@@@@@@@@@@@@@@@@ Test:  ',len(i), ' ; ',len(j) )
              #print(i)
          elif transLen > self.max_length:
              j = i
              del j[self.max_length:]
              transX.append(j)
          else:
              transX.append(i)

          #transformedData[ind] = [worddict[w] if w in worddict else 1 for w in words]
       #print('############################# Transformed Data ###################')
       #print(transX)
       return (transX, transformedDataY)
    
    def split_Train_Validate(self,tx,ty,validatePercent):
        validX = []
        validY = []
        trainX = []
        trainY = []
        n_samples = len(tx)
        sidx = numpy.random.permutation(n_samples)
        n_train = int(numpy.round(n_samples * (1. - validatePercent)))
        validX = [tx[s] for s in sidx[n_train:]]
        validY = [ty[s] for s in sidx[n_train:]]
        trainX = [tx[s] for s in sidx[:n_train]]
        trainY = [ty[s] for s in sidx[:n_train]]

        print('********************** Validation Data *********************')
        print(len(validX), ' ; ' , len(validY))
        print('********************** Training Data ***********************')
        print(len(trainX), ' ; ', len(trainY))

        return ((trainX, trainY),(validX,validY))


    def prepareData(self,dataX):
       wordList=[]
       for i in xrange(0,len(dataX)):
          wordList[i] = re.sub("[^\w]", " ",  dataX[i]).split()
          #print(dataX[i],' :::::::::::::::::::::   ',wordList[i])       
    '''
   
def main():
   print('Initializing the LSTM Model')
   lstm = LSTMSentiment()
   
   print('Retrieving the Training and Test Data')
   path = os.getcwd()
   ((TrainX,TrainY),(TestX,TestY)) = lstm.getTrainTestData()
       
   '''
   worddict = dict()
   worddict = lstm.build_dict(trainX,testX)

   print('Transforming Training and Test Data')
   (TrainX,TrainY) = lstm.transformData(trainX,trainY,worddict)
   
   ((TrainX,TrainY),(ValidX,ValidY)) = lstm.split_Train_Validate(TrainX,TrainY,0.1)

   print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
   #print(len(set([len(a) for a in TrainX] + [len(TrainY)])) )
   print([len(a) for a in TrainX] + [len(TrainY)])
   print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
   print(len(set([len(a) for a in ValidX] + [len(ValidY)])) )


    
   print('############################ TRain and TEst Data ######################')
   print(len(TrainX),' : ',len(TrainY), ': : ',len(TestX),' : ',len(TestY))
   print(TestX)
   print(TestY)

   

   (TestX,TestY) = lstm.transformData(testX,testY,worddict)

   TrainX = numpy.array(TrainX)
   TrainY = numpy.array(TrainY)
   TrainY = TrainY.reshape(TrainY.shape[0],1)
   
   print('************* After Numpy transformation : Training Data  *****************')
   #print(TrainX)
   print(TrainX.shape)
   print('--------------------------------')
   #print(TrainY)
   print(TrainY.shape)
   


   ValidX = numpy.array(ValidX)
   ValidY = numpy.array(ValidY)
   ValidY = ValidY.reshape(ValidY.shape[0],1)  

   print('************* After Numpy transformation : Validation Data  *****************')
   #print(TrainX)
   print(ValidX.shape)
   print('--------------------------------')
   #print(TrainY)
   print(ValidY.shape)   


   TestX = numpy.array(TestX)
   TestY = numpy.array(TestY)
   TestY = TestY.reshape(TestY.shape[0],1)
   
   
   print('************* After Numpy transformation *****************')
   #print(TestX)
   print(TestX.shape)
   print('--------------------------------')
   #print(TestY)
   print(TestY.shape)
   '''
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
   #print('+++++++++++++++++++++++++++++++++++')
   #print(DataX, ': : ', DataY)
   lstm.predictSentiment(DataX)


   
if __name__ =='__main__':
   main()
