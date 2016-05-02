import SpeechNetSVMMulticlass as sp
import SVMSentimentAnalysis_Spark as l
import speech_recognition as sr
import datetime
import sys

def main(filename):
    # Initialize Speech Recognition Algorithm
    r = sr.Recognizer()
    indatetime = datetime.datetime.now().time().isoformat()
    # Setting the path of the Speech Recognition Output
    textfilename = "./Data/"+str(indatetime).replace(":","").replace(".","")+".txt"

    with sr.WavFile(filename) as source:
        audioData = r.record(source)                                                #Converting the wav file into Audio Data file
        textData = r.recognize_sphinx(audio_data=audioData, language="en-US")       # Setting the language as English
        f = open(textfilename, "wb")
        f.write(textData)
        f.close()

    # Invoking the Emotion recognition module
    emotions = sp.main(filename)
    # Invoking Text to emotion recognition module
    annotation = l.main(textfilename)

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
    return final_emotion_data

if __name__ == '__main__':
    if len(sys.argv) == 1:
        main("./Data/LDC93S1.wav")
    else:
        main(sys.argv[1])

