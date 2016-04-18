import numpy as np
import scipy.io.wavfile as sc
import scipy.fftpack as sc1
import matplotlib.pyplot as plt



freq,input= sc.read("/media/vyassu/OS/Users/vyas/Documents/Assigments/BigData/AudioData/DC/d01.wav")
print len(input)
inputfft = sc1.fft(input)
print inputfft.imag





inputfftList=list(inputfft)

maxvalue= inputfftList.index(max(inputfft))

print maxvalue

inputlog = np.log(inputfft)

newInput = sc1.fft(inputlog)

newAbsInput= np.abs(newInput[10:len(newInput)-100])

print newAbsInput

newAbsInputList = list(newAbsInput)

maxValue = max(newAbsInput)
maxValueIndex = newAbsInputList.index(maxValue)

print maxValue,maxValueIndex,freq
sc.write("/media/vyassu/OS/Users/vyas/Documents/Assigments/BigData/output.wav",rate=freq,data=np.asarray(newAbsInput,dtype=np.int16))

plt.figure(1)
plt.title('Input Signal')
plt.plot(input)

plt.figure(2)
plt.title('after filtering Signal')
plt.plot(np.fft.ifft(inputfft))

plt.figure(6)
plt.title('after filtering Signal')
plt.ylim([0,maxValue])
plt.plot(newAbsInput)

plt.show()