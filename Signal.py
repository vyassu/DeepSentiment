import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile as sc
import speech_recognition as sr
import glob


class signal:

    def convertTO1DArray(self,input):
        length = len(input[0])

        finalArray=[]

        for elementList in input:
            print elementList
            for element in list(elementList):
                temp=[]
                i = 0
                if len(finalArray) < len(elementList):
                    temp.append(element)
                    finalArray.append(temp)
                else:
                    finalArray[i]=finalArray[i] + element
                    print finalArray
                i=i+1
        return input

    def getSample(self,input):
        sum=0.0
        for i in range(0,200):
            sum=sum+input[i]
        avg=sum/200
        temp,newtemp,newtemp1=[],[],[]
        for i in range(0,len(input)):
            temp.append(int(input[i]-avg))
        maxValue = sorted(temp,reverse=True)[0]
        maxpos=0
        for i in range(0,len(input)):
            if temp[i]==maxValue:
                maxpos=i

        Eb=maxValue/maxpos
        Es=maxValue/len(input)-maxpos
        Sb=0
        Ss=0

        for i in range(0,len(temp),350):
            sum=0
            temp1=temp[i:350+i]
            for element in temp1:
                sum=sum+element
            newtemp.append(float(sum/350))

        for i in range(0,len(newtemp)-1):
            if newtemp[i]!=0:
                newtemp1.append(newtemp[i+1]/newtemp[i])
            else:
                newtemp1.append(0.0)

        for i in range(0,len(newtemp1)):
            if newtemp1[i]>Eb:
                Sb=i*350

        for i in range(len(newtemp1),0,-1):
            if newtemp1[i-1]!=0:
                if(1/newtemp1[i-1])>Es:
                    Ss=i*350
        return temp[Ss:Sb]

    def phonemes(self,input):

        temp1,newtemp,newtemp1,a=[],[],[],[]
        for i in range(0,len(input),500):
            sum=0
            temp1=input[i:500+i]
            for element in temp1:
                sum=sum+element
            newtemp.append(float(sum/500))

        for i in range(0,len(newtemp)-1):
            if newtemp[i]!=0:
                newtemp1.append(newtemp[i+1]/newtemp[i])
            else:
                newtemp1.append(0.0)

        for i in range(0,len(newtemp1)-1):
            if str(newtemp1[i])=='-0.0':
                a.append(-1)
            elif newtemp1[i]>=0:
                a.append(1)
            else:
                a.append(-1)
        endVector=[]
        for i in range(0,len(a)-1):
            if a[i-1]==-1 and a[i]==-1:
                endVector.append(500*i)
        syllables=[]
        for i in range(0,len(endVector)-1):
            if i==0:
                syllables.append(input[0:endVector[0]])
            else:
                syllables.append(input[endVector[i-1]:endVector[i]])

        return syllables

    def getsyllab(self,path):
        inFreq,input = sc.read(path)
        test = self.getSample(input)
        syllab = self.phonemes(test)
        return syllab,inFreq

    def getvals(self,path):
        finalOutput = []
        inFreq,input = sc.read(path)
        test = self.getSample(input)
        syllab = self.phonemes(test)
        finalOutput.append(str(input))
        for element in syllab:
            finalOutput.append(str(element))
        return finalOutput

if __name__=="__main__":
    HOME="/media/vyassu/OS/Users/vyas/Documents/Assigments/BigData/"
    sig=signal()
    r = sr.Recognizer()
    syllab,inFreq = sig.getsyllab(HOME+"NewDataset/sp01.wav")
    for i in range(0,len(syllab)):
        filename = HOME+"syllab/syllab"+str(i)+".wav"
        sc.write(filename ,rate=inFreq,data=np.asarray(syllab[i],dtype=np.int16))
        with sr.WavFile(filename) as source:              # use "test.wav" as the audio source
            audio = r.record(source)                        # extract audio data from the file
            try:
                print("Transcription: " + r.recognize_sphinx(audio_data=audio,language="en-US") )  # recognize speech using Google Speech Recognition
            except LookupError:                                 # speech is unintelligible
                print("Could not understand audio")
            except :
                print ("UNknow error")


#for f in gb.glob("/media/vyassu/OS/Users/vyas/Documents/Assigments/BigData/NewDataset/*.wav"):
    #     finalOutput = []
    #     inFreq,input = sc.read(f)
    #     test = getSample(input)
    #     syllab = phonemes(test)
    #     finalOutput.append(list(input))
    #     finalOutput.append(list(syllab))
    #     filename="/media/vyassu/OS/Users/vyas/Documents/Assigments/BigData/Output/temp"+str(i)+".txt"
    #     np.savetxt(filename,np.asarray(finalOutput),fmt="%s")
    #     i=i+1



# plt.plot(input)
# for i in range(0,len(syllab)):
#     plt.figure(i+2)
#     sc.write("/media/vyassu/OS/Users/vyas/Documents/Assigments/BigData/syllab"+str(i)+".wav",rate=inFreq,data=np.asarray(syllab[i],dtype=np.int16))
#     plt.plot(np.array(syllab[i]))
# plt.show()

