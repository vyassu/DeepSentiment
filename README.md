# DeepSentiment
Speech Emotion Recognition using Fast Fourier Transform and Support Vector Machine

This module aims at extracting emotion components within human speech like Pitch and Loudness. These parameters are used to predict the emotion state of the speaker in question.

## Prerequisites
1.) The following are the prerequsite python modules that needs to be installed to execute the Standalone component:
```
sudo pip install numpy 
sudo pip install scipy
sudo pip install pandas
sudo pip install SpeechRecognition
sudo pip install -U scikit-learn
```
2.) Follow the instructions mentioned in the [ link ](http://aubio.org/) to install Aubio(pitch extraction library).

3.) If you want to train your own model, then install the latest version of [ Apache Spark ] (http://spark.apache.org/downloads.html)
