import os,glob

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
    sentences =[]
    trainX=[]
    trainY=[]

    # Extracting the Positive sentiments
    posFilePath = os.path.join(filePath,'pos/')
    posFiles = os.listdir(posFilePath)
    for i in posFiles:
      if i.endswith('.txt') :
         f = open(posFilePath+'/'+i,'r')
         sentences.append(f.readline().strip())
         trainX.append(f.readline().strip())
         trainY.append('1')
    f.close()

    # Extracting the Negative sentiments
    negFilePath = os.path.join(filePath,'pos/')
    negFiles = os.listdir(negFilePath)
    for i in negFiles:
      if i.endswith('.txt') :
         f = open(negFilePath+'/'+i,'r')
         sentences.append(f.readline().strip())
         trainX.append(f.readline().strip())
         trainY.append('0')
    f.close()
    
    print('*****************  Test ***************')
    print(type(sentences), sentences[0])
    print(sentences[1])
    print(len(sentences))


    return (trainX,trainY)
    

def main():
   global DIRECTORY
   preProc = preProcessor()
   path = os.getcwd() # This needs to be executed from within the DIRECTORY
   
   # Get the training data from the 'train' folder
   print('Extracting the Training Data')
   (trainX,trainY) = preProc.train_test_split(path,'train')

 
   # Get the test data from the 'test' folder
   print('Extracting the Test data')
   (testX,testY)  = preProc.train_test_split(path,'test')

   #preProc.getData(DIRECTORY,'train')


if __name__ =='__main__':
   main()
