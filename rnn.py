from __future__ import print_function

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import LSTM
from keras.optimizers import SGD
from pyspark import SparkContext

import sys
import numpy as np
import pandas as pd
import LSTM_RNN as ls

from elephas.utils.rdd_utils import to_simple_rdd
from elephas.spark_model import SparkModel
from elephas import optimizers as elephas_optimizers
import elephas.spark_model as sm 


def load_data(data, label, n_prev = 30):
   """
   data should be pd.DataFrame()
   """
   n_prev=n_prev
        
   docX, docY = [],[]
   for i in range(len(data)-n_prev):
      docX.append(data.iloc[i+n_prev].as_matrix()) #i:
      docY.append(label.iloc[i+n_prev].as_matrix())
   alsX = np.array(docX)
   alsY = np.array(docY)
   return alsX, alsY


def train_test_split(df, label, test_size=0.2):
   ntrn = int(round(len(df) * (1 - test_size)))
   print('Number of training sample', ntrn)
   print('Number of testing sample', len(df) - ntrn)

   x_train, y_train = load_data(df.iloc[0:ntrn], label.iloc[0:ntrn])

   x_test, y_test = load_data(df.iloc[ntrn:], label.iloc[ntrn:],0)  

   '''
   print('$$$$$$$$$$$$$$$$$$$$$$$$$$$  Test Data inside train_test_split method')
   print('Training Data')
   print(x_train)
   print('-------------------')
   print(y_train)

   print('Test data')
   print(df.iloc[ntrn:])
   print('--------------------')
   print(label.iloc[ntrn:])

   print('Test data : x_test')
   print(x_test)
   print('Test Data : y_test')
   print(y_test)
   '''
   return (x_train, y_train), (x_test, y_test)

if __name__=="__main__":
   if(len(sys.argv)!=2):
     print("Fatal: incorrect number of arguments, Usage : train.py <dataset>")
     exit(-1)

   sc = SparkContext(appName = "TrainPhonemes")

   hidden_neurons = 300
   in_dim = 19
   out_dim = 1   

   # Reading the Input and the expected Output from the dataset
   testinput = pd.read_csv(filepath_or_buffer=str(sys.argv[1]),usecols=['Phoneme'])
   testoutput = pd.read_csv(filepath_or_buffer=str(sys.argv[1]),usecols=['Word'])
  
   print('########################## testinput',testinput)
   print('########################## testoutput',testoutput)   
      
   print('Initialising the Keras Sequential Model')
   model = Sequential()

   print('Adding layers')
   model.add(LSTM(output_dim=hidden_neurons,input_dim = in_dim,return_sequences=False))
   model.add(Dropout(0.5))
   model.add(Dense(output_dim=out_dim,input_dim=300,activation='tanh'))
   model.compile(loss="categorical_crossentropy", optimizer='sgd',class_mode="binary")
   
   print('Creating Training and Test Data')
   ((x_train,y_train),(x_test,y_test)) = train_test_split(testinput.fillna(0),testoutput.fillna(0), test_size=0.3)
   
   print('Training data : x')
   print(type(x_train))
   print(x_train)
   print('Training data : y')
   print(type(y_train))
   print(y_train)   
 
   print('Test data : x')
   print(type(x_test))
   print(x_test)
   print('Test data : y')
   print(type(y_test))
   print(y_test)   
   
   print('Converting training data to RDD')
   rddataset = to_simple_rdd(sc, x_train, y_train)
   
   print('Initializing SPark Model')
   sgd = elephas_optimizers.SGD()
   spark_model = SparkModel(sc,model,optimizer=sgd ,frequency='epoch', mode='asynchronous', num_workers=2)

   print('Commencing training')
   spark_model.train(rddataset, nb_epoch=10, batch_size=200, verbose=1, validation_split=0)  
   #model.fit(x_train, y_train, nb_epoch=5, batch_size=32) 
   print('Training completed')  


   


   sc.stop()
