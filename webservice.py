from flask import Flask,make_response,request
from flask_cors import CORS, cross_origin
import numpy as np
import scipy.io.wavfile as sc
import json
import base64
import Controller
import datetime,struct
senti = Flask(__name__)
CORS(senti)

def convert_float32_wav_file(data,samplerate):
    byte_count = (len(data)) * 4
    wav_file = ""
    wav_file += struct.pack('<ccccIccccccccIHHIIHH',
                            'R', 'I', 'F', 'F',
                            byte_count + 0x2c - 8,  # header size
                            'W', 'A', 'V', 'E', 'f', 'm', 't', ' ',
                            0x10,3,1,samplerate,samplerate*4,4,32)
    wav_file += struct.pack('<ccccI','d', 'a', 't', 'a', byte_count)
    for dat in data:
        wav_file += struct.pack("<f", dat)
    return wav_file

@senti.route('/deepsentimentweb', methods=['GET', 'POST'])
def webservice():
    inputdatetime = datetime.datetime.now().time().isoformat()
    absfile = "./Data/"+str(inputdatetime).replace(":","").replace(".","")+".wav"

    datarequest = json.loads(request.data)["data"]
    data= json.loads(datarequest)
    samplerate = json.loads(request.data)["samplerate"]

    ##sc.write(absfile, rate=samplerate,data=np.asarray(data, dtype=np.float32))
    filecontents = convert_float32_wav_file(data,samplerate)

    with open(absfile,"wb") as f:
        for data in filecontents:
            f.write(data)
    f.close()
    resultData = Controller.main(absfile)
    print resultData

    emotion = "You are feeling "
    i = 0
    for key in resultData.keys():
        if key != "data":
            if ((resultData[key] == 1.0)):
                if i < len(resultData.keys()) - 1:
                    emotion = emotion + key + " "
                else:
                    emotion = emotion + "OR " + key + "."
        i = i + 1;
    finalData = {}
    finalData["input"] = resultData["data"]
    finalData["result"] = emotion
    resp = make_response(json.dumps(finalData))
    resp.headers['Content-Type'] = "application/json"
    resp.headers['Access-Control-Allow-Origin'] = "*"
    return resp

@senti.route('/deepsentifile', methods=['GET', 'POST'])
def fileservice():
    inputdatetime = datetime.datetime.now().time().isoformat()
    absfile = "./Data/" + str(inputdatetime).replace(":", "").replace(".", "") + ".wav"

    soundata = request.data
    soundata = base64.b64decode(soundata[soundata.index(",")+1:])

    with open(absfile, "wb") as f:
        f.write(soundata)
    f.close()
    resultData = Controller.main(absfile)
    print resultData

    emotion = "You are feeling "
    i=0
    for key in resultData.keys():
        if key != "data":
            if ((resultData[key]==1.0)) :
               if i < len(resultData.keys())-1:
                    emotion= emotion+key+" "
               else:
                    emotion = emotion+"OR "+key+"."
        i=i+1;
    finalData ={}
    finalData["input"] = resultData["data"]
    finalData["result"] = emotion
    resp = make_response(json.dumps(finalData))
    resp.headers['Content-Type'] = "application/json"
    resp.headers['Access-Control-Allow-Origin'] = "*"
    return resp

def main():
    senti.run(debug=True)

if __name__ == "__main__":
    main()