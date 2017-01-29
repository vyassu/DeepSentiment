import SpeechNetSVMMulticlass as sp
import SVMSentimentAnalysis_Spark as l
import speech_recognition as sr
import datetime,sys
import logging
py_ver = sys.version_info[0]
readinput = ""
'''
    CRITICAL 	50
    ERROR 	40
    WARNING 	30
    INFO 	20
    DEBUG 	10
    NOTSET 	0
'''
logging_level = 20

logging.getLogger("Controller")
logging.basicConfig(level=logging_level)

if py_ver == "3":
    readinput = "input()"
else:
    readinput = "raw_input()"

def main(filename,training=False,*args):
    # Initialize Speech Recognition Algorithm
    logging.info("Loaded the Speech Recognition Module")
    r = sr.Recognizer()
    indatetime = datetime.datetime.now().time().isoformat()
    # Setting the path of the Speech Recognition Output
    textfilename = "./Data/"+str(indatetime).replace(":","").replace(".","")+".txt"
    logging.info("Saved the input signal into WAVE File")
    with sr.WavFile(filename) as source:
        audioData = r.record(source)                                                #Converting the wav file into Audio Data file
        textData = r.recognize_sphinx(audio_data=audioData, language="en-US")       # Setting the language as English
        f = open(textfilename, "wb")
        f.write(textData)
        f.close()
    logging.info("Converted Speech To Text")
    # Invoking the Emotion recognition module
    if training == True:
        logging.info("Training of Model Enabled")
        emotions = sp.main(filename,training,logging_level,args[0])
        # Invoking Text to emotion recognition module
        annotation = l.main(textfilename,training,logging_level,args[1:])
    else:
        logging.info("Training of Model Disabled")
        emotions = sp.main(filename, training,logging_level)
        # Invoking Text to emotion recognition module
        annotation = l.main(textfilename, training,logging_level)

    positive_emotions = ["happy", "surprise"]                                       # Variable to store positive emotions
    negative_emotions = ["Angry", "Disgust", "surprise", "sadness", "fear"]         # Variable to store negative emotions
    final_emotion_data = {}                                                         # Variable to store Final emotion set

    # Condition to check whether the emotions are neutral
    if annotation[0] == '1':
        for emotion in positive_emotions:
            if len(emotions.get(emotion)) != 0:
                final_emotion_data.update({emotion: emotions.get(emotion)})
    else:
        for emotion in negative_emotions:
            if emotions.get(emotion) != None:
                final_emotion_data.update({emotion: emotions.get(emotion)})

    # Condition to check whether the emotions are neutral
    if len(final_emotion_data) == 0:
        final_emotion_data.update({"Neutral": 1.0})
    final_emotion_data.update({"data":textData})
    logging.info("Final Emotion set:: %s",str(final_emotion_data))
    return final_emotion_data

if __name__ == '__main__':
    print("    WELCOME TO DEEPSENTIMENT     ")
    print(" 1. TEST using .wav file")
    print(" 2. TRAIN-TEST Your own model")
    userinput = eval(readinput)
    print("Enter WAVE Filepath")
    inputpath = eval(readinput)
    if userinput == "1":
        print main(inputpath, False) #print main("./Data/a01.wav",False)
    else:
        print("Enter Audio(WAVE File) test-train data filepath")
        audiodatapath = eval(readinput)
        print("Enter Positive Train data filepath")
        postrainpath = eval(readinput)
        print("Enter Negative Train data filepath")
        negtrainpath = eval(readinput)
        print("Enter Positive Test data filepath")
        postestpath = eval(readinput)
        print("Enter Negative Test data filepath")
        negtestpath = eval(readinput)
        print main(inputpath,True,audiodatapath,postrainpath,negtrainpath,postestpath,negtestpath)

