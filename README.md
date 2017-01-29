# DeepSentiment
Speech Emotion Recognition using Fast Fourier Transform and Support Vector Machine

This module aims at extracting emotion components within human speech like Pitch and Loudness and use them to identify the emotion state of the speaker. Support Vector Machines are used to segregate the features into various emotion states like Anger, Sadness, Fear, Happy and Neutral. Some of these emotion states are interleaved, reducing the precision with which we can decipher the emotion state, hence we have also incorporated text based sentiment recognition to improve precision of prediction. We have used Pyspark (Apache Spark) library to develop the model for this purpose.

## Prerequisites
1.) The following are the prerequsite python modules that needs to be installed to execute the Standalone component:
```
sudo pip install numpy 
sudo pip install scipy
sudo pip install pandas
sudo pip install SpeechRecognition
sudo pip install -U scikit-learn
sudo pip install findspark
sudo pip install flask
sudo pip install flask_cors
```
Note: There may be other prerequiste library files that needs to installed before installing the above mentioned modules.

2.) Follow the instructions mentioned in the [ link ](http://aubio.org/) to install Aubio(pitch extraction library).

3.) If you want to train your own model, then install the latest version of [ Apache Spark ] (http://spark.apache.org/downloads.html) and use the code inside Spark for training the model.

## Downloads
Clone the repository using the below mentioned command and execute the bash script.
```
git clone https://github.com/vyassu/DeepSentiment.git
cd DeepSentimemt/Code/StandAlone
chmod 755 script.sh
$./script.sh
```

## Test and Run

There are two ways to run the program

1.) HTML/CSS userinterface through which you can record your voice and get the output, or upload a WAV file. in your browser paste the below 
```
URL http://localhost:5000/deepsentiment
```
2.) Execute the below mentioned command 
```
      python Controller.py
```
Follow the directions for commandline testing.

##Note: The record voice feature is still in development stage!!
