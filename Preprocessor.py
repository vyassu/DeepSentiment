import os
import cPickle

DIRECTORY = "aclImdb"

class preProcessor:

  def __init__(self):
    print('Initializing preprocessor')


  def getData(self, datasetPath , dataType):
    print('Test')
    filepath = os.path.join(datasetPath,'train/')
    files = glob.glob(filepath)
    print(type(files),'    ',files)


  def train_test_split(self,datasetpath,datatype):
 
    filePath = os.path.join(datasetpath,datatype+'/')
    print('Filepath  ' , filePath)
    sentences =[]
    dataX=[]
    dataY=[]

    # Extracting the Positive sentiments
    posFilePath = os.path.join(filePath,'pos/')
    print('Positive :  ',posFilePath)
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

    
    '''
    print('*****************  Test ***************')
    print(type(sentences), sentences[0])
    print(sentences[1])
    print(len(sentences))
    '''

    return (dataX,dataY)
    

def main():
   global DIRECTORY
   preProc = preProcessor()
   path = os.getcwd() # This needs to be executed from within the DIRECTORY
   
   # Get the training data from the 'train' folder
   print('Extracting the Training Data')
   (trainX,trainY) = preProc.train_test_split(path,'train')
   '''
   print('******************************* Training Data ***************************')
   for i in xrange(0,len(trainX)):
     print(trainX[i] ,':::::::::' ,trainY[i])
   '''
 
   # Get the test data from the 'test' folder
   print('Extracting the Test data')
   (testX,testY)  = preProc.train_test_split(path,'test')
   '''
   print('******************************* Test Data ***************************')
   print(len(testX) , '....',len(testY))
   for i in xrange(0,len(testX)):
     print(testX[i] ,':::::::::' ,testY[i])
   '''

   # Dumping the Training and Test data
   trainDump = open("trainingdata.pkl","wb")
   testDump = open("testingdata.pkl","wb")
   print('Dumping the Training Data')
   cPickle.dump((trainX,trainY),trainDump)
   print('Dumping the Test Data')
   cPickle.dump((testX,testY),testDump)
   
   trainDump.close()
   testDump.close()

if __name__ =='__main__':
   main()
