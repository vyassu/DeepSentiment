import SpeechNetSVMMulticlass as sp
import LSTMSentiment as l

if __name__ == '__main__':
    emotions = sp.main()
    annotation =  l.main()

    positive_emotions = ["happy","surprise"]
    negative_emotions = ["Angry","Disgust","surprise","sadness","fear"]
    final_emotion_data = {}
    if annotation[0]=='1':
        for emotion in positive_emotions:
            if emotions.get(emotion) != None:
                final_emotion_data.update({emotion:emotions.get(emotion)})
    else:
        for emotion in negative_emotions:
            if emotions.get(emotion)!=None:
                final_emotion_data.update({emotion: emotions.get(emotion)})
    print final_emotion_data

