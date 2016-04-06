from aubio import source
from numpy import array,ma
import aubio as au

def getPitch(filename):
    from aubio import pitch
    downsample = 1
    samplerate = 44100 / downsample
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

    return max(cleaned_pitches)

if __name__=="__main__":
    print getPitch("/media/vyassu/OS/Users/vyas/Documents/Assigments/BigData/AudioData/DC/a12.wav")
    print getPitch("/media/vyassu/OS/Users/vyas/Documents/Assigments/BigData/AudioData/DC/d12.wav")
    print getPitch("/media/vyassu/OS/Users/vyas/Documents/Assigments/BigData/AudioData/DC/su12.wav")

