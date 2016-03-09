import numpy as np
import scipy as sc
import wave as w
import matplotlib.pyplot as plt
import sys

def getHighPassFilter():
    a=[0.48296, -0.83651, 0.22414, 0.12940]
    H3=[a[2]+a[0], a[3]+a[1]]
    H2=[a,[a[2],a[3],a[0],a[1]]]
    H1=[]
    return H3,H2

def getLowPassFilter():
    a=[-0.12940, 0.22414, 0.83651, 0.48296]
    L3=[a[2]+a[0], a[3]+a[1]]
    L2=[a,[a[2],a[3],a[0],a[1]]]
    L1=[]
    return L3,L2


H3,H2 = getHighPassFilter()

readfile=w.open("/media/vyassu/OS/Users/vyas/Documents/Assigments/BigData/LDC93S1.wav","r")
input = readfile.readframes(readfile.getnframes())
output1 = []

signal = np.fromstring(input, 'Int16')

for i in range(0,readfile.getnframes()):
     print int(input[i].encode("hex"),16)
     temp = float(H3[0])* int(input[i].encode("hex"),16)
     output1.append(temp.hex())


#If Stereo
if readfile.getnchannels() == 2:
    print 'Just mono files'
    sys.exit(0)

outputsignal = np.fromstring(str(output1),'Int16')

plt.figure(1)
plt.title('Signal Wave...')
plt.plot(signal)

plt.figure(2)
plt.title('Output Signal Wave...')
plt.plot(outputsignal)


plt.show()
#print readfile.readframes(readfile.getnframes())



