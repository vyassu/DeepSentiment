from aubio import source
from numpy import array,ma
import aubio as au

def getPitch(filename,frate):
    from aubio import pitch
    downsample = 1
    samplerate = frate / downsample
    win_s = 1024 / downsample # fft size

    hop_s = 256/ downsample # hop size
    s = source(filename, samplerate, hop_s)
    samplerate = s.samplerate
    tolerance = 0.8

    pitch_o = pitch("yin", win_s, hop_s, samplerate)
    pitch_o.set_unit("midi")
    pitch_o.set_tolerance(tolerance)
    total_frames = 0
    pitches = []
    confidences=[]
    while True:
        samples, read = s()
        pitch = pitch_o(samples)[0]
        confidence = pitch_o.get_confidence()
        confidences+=[confidence]
        pitches += [pitch]
        total_frames += read
        if read < hop_s: break
    pitches = array(pitches[1:])
    confidences = array(confidences[1:])
    cleaned_pitches = pitches
    cleaned_pitches = ma.masked_where(confidences < tolerance, cleaned_pitches,copy=False)
    cleaned_pitches = cleaned_pitches[~cleaned_pitches.mask]

    print cleaned_pitches
    if len(cleaned_pitches)==0:
        maxValue = 0
    else:
        maxValue = max(cleaned_pitches)
    return maxValue

if __name__=="__main__":
    print getPitch("/home/vyassu/PycharmProjects/DeepSentiment/Data/sp04.wav",8000)
    #print getPitch("/media/vyassu/OS/Users/vyas/Documents/Assigments/BigData/AudioData/DC/d12.wav")
    #print getPitch("/media/vyassu/OS/Users/vyas/Documents/Assigments/BigData/AudioData/DC/su12.wav")

