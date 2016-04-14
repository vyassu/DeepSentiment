import os
import cPickle
import numpy
import re

max_length  = 500

def getData(datasetPath , dataType):
    print('Test')
    filepath = os.path.join(datasetPath,'train/')
    files = glob.glob(filepath)
    print(type(files),'    ',files)


def train_test_split(datasetpath,datatype):
    filePath = os.path.join(datasetpath,datatype+'/')
    print('Filepath  ' , filePath)
    sentences =[]
    dataX=[]
    dataY=[]

    # Extracting the Positive sentiments
    posFilePath = os.path.join(filePath,'pos/')
    posFiles = os.listdir(posFilePath)
    for i in posFiles:
      if i.endswith('.txt') :
         f = open(posFilePath+'/'+i,'r')
         #sentences.append(f.read())
         dataX.append(f.read())
         dataY.append('1')
         f.close()

    # Extracting the Negative sentiments
    negFilePath = os.path.join(filePath,'neg/')
    negFiles = os.listdir(negFilePath)
    for i in negFiles:
      if i.endswith('.txt') :
         g = open(negFilePath+'/'+i,'r')
         #sentences.append(f.read())
         dataX.append(g.read())
         dataY.append('0')
         g.close()

    return (dataX,dataY)
    
  # This method builds a dictionary of words combining the training and test data which is used during the sentence to integer vector transformation
def build_dict(trainX,testX):
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
       print('@@@@@@@@@@@@@@@@@@@@@@@@@@@ Sorted ')
       print(sorted_idx)

       for idx, ss in enumerate(sorted_idx):
          worddict[keys[ss]] = idx+2
    
       return worddict

  # Transforms sentences into Integer vectors where number represents value corresponding to the word in the Dictionary built above
def transformData(dataX,dataY,worddict):
       global max_length
       transX = []
       transformedDataY = dataY
       transformedDataX = [None] * len(dataX)
       for i in xrange(len(dataX)):
          transformedDataX[i]=[]
          dataX[i]= dataX[i].replace(',','')
          dataX[i]= dataX[i].replace('_','')
          words = re.sub("[^\w]", " ", dataX[i]).lower().split()
          for w in words:
            transformedDataX[i].append(int(w,36))
          '''
          print('-------------------------------')
          print(len(words), ' : ',words)
          print('###############################')
          print(type(transformedDataX), ' : ', len(transformedDataX))
          '''

       for i in transformedDataX:
          transLen = len(i)
          if(transLen < max_length): #Pad zeroes to the data vector
              transX.append(i+[0]*(max_length - transLen))
          elif transLen > max_length:
              j = i
              del j[max_length:]
              transX.append(j)
          #transformedData[ind] = [worddict[w] if w in worddict else 1 for w in words]
          else:
              transX.append(i)
       transX = numpy.array(transX)
       transformedDataY = numpy.array(transformedDataY)
       transformedDataY = transformedDataY.reshape(transformedDataY.shape[0],1)
       '''
       print('###############################')
       print(type(transX), ' : ', len(transX))
       print(transX)
       
       global max_length
       transformedDataX = [None] * len(dataX)

       for ind,sen in enumerate(dataX):
          words = re.sub("[^\w]", " ", sen).lower().split()
          transformedDataX[ind]=[]
          for w in words:
             if w in worddict:
                transformedDataX[ind].append(worddict[w])
             else:
                transformedDataX[ind].append(1)
          
       #Converting the length of the transformed data to maximum length

       for i in transformedDataX:
          transLen = len(i)
          if(transLen < max_length): #Pad zeroes to the data vector
              transX.append([0]*(max_length - transLen) + i)
          elif transLen > max_length:
              j = i
              del j[max_length:]
              transX.append(j)
          #transformedData[ind] = [worddict[w] if w in worddict else 1 for w in words]
          else:
              transX.append(i)
       transX = numpy.array(transX)
       transformedDataY = numpy.array(transformedDataY)
       transformedDataY = transformedDataY.reshape(transformedDataY.shape[0],1)
       '''
       return (transX, transformedDataY)


def main():
   #preProc = preProcessor()
   path = os.getcwd() # This needs to be executed from within the DIRECTORY
   
   # Get the training data from the 'train' folder
   print('Extracting the Training Data')
   (trainX,trainY) = train_test_split(path,'train')

   # Get the test data from the 'test' folder
   print('Extracting the Test data')
   (testX,testY)  = train_test_split(path,'test')

   print('Building the dictionary')
   worddict = dict()
   worddict = build_dict(trainX,testX)

   print('Transforming Training and Test Data')
   (TrainX,TrainY) = transformData(trainX,trainY,worddict)
   '''
   (TestX,TestY) = transformData(testX,testY,worddict)

   # Dumping the dictionary to be used for external speech text input
   dictDump = open("dictionary.pkl","wb")
   print('Dumping the dictionary data')
   cPickle.dump(worddict,dictDump,-1)

   # Dumping the Training and Test data
   trainDump = open("trainingdata.pkl","wb")
   testDump = open("testingdata.pkl","wb")
   print('Dumping the Training Data')
   cPickle.dump((TrainX,TrainY),trainDump,-1)
   trainDump.close()

   print('Dumping the Test Data')
   cPickle.dump((TestX,TestY),testDump,-1)
   testDump.close()
   '''

   return 'Success'

if __name__ =='__main__':
   main()
