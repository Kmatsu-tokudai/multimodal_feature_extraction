# SpeechBrainによる音声感情分析
import sys, os, re, glob
import pandas as pd
import numpy as np
import wave
# SpeechBrain
#import speechbrain as sb
from speechbrain.inference.interfaces import foreign_class

classifier = foreign_class(source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP", 
            pymodule_file="custom_interface.py", classname="CustomEncoderWav2vec2Classifier")


def toStr(vec):
    s = ''
    for v in vec:
        s += f'{v:.3f} '
    s = s.rstrip(' ')
    return s

# SpeechBrainで音声感情分析
def doSpeechBrain(speech_path):
    out_prob, score, index, text_lab = classifier.classify_file(speech_path)
    print(out_prob, score, index, text_lab, sep='\t')
    ov = out_prob.to('cpu').detach().numpy().copy()
    scoreV = score.to('cpu').detach().numpy().copy()
    indexV = index.to('cpu').detach().numpy().copy()
    return ov, scoreV[0], indexV[0]

def estSpeechBrain(speech_path):
    # WAVファイルを開く
    ovx = np.array([0.0] * 4)
    ch = {'hap':0.0, 'ang':0.0, 'sad':0.0, 'neu':0.0}
    num = 0.0
    with wave.open(speech_path, 'rb') as wav_file:
        frame_rate = wav_file.getframerate()
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        frame_count = wav_file.getnframes()
        
        # 10秒ごとにファイルを区切る
        segment_length = 10 * frame_rate
        
        # WAVファイルを読み込む
        for i in range(0, frame_count, segment_length):
            temp_file_name = 'temp.wav'
            # WAVファイルを書き込む
            with wave.open(temp_file_name, 'wb') as new_wav_file:
                # WAVファイルのヘッダーを設定
                new_wav_file.setframerate(frame_rate)
                new_wav_file.setnchannels(channels)
                new_wav_file.setsampwidth(sample_width)
                # WAVファイルからデータを読み込む
                segment = wav_file.readframes(segment_length)
                # WAVファイルにデータを書き込む
                new_wav_file.writeframes(segment)

            # 感情認識
            try:
                out_prob, score, index, text_lab = classifier.classify_file(temp_file_name)
            except:
                continue
            
            #print(out_prob, score, index, text_lab, sep='\t')
            ov = out_prob.to('cpu').detach().numpy().copy()
            scoreV = score.to('cpu').detach().numpy().copy()
            indexV = index.to('cpu').detach().numpy().copy()
            ch[text_lab[0]]+=1.0
            ovx += ov[0]
            num += 1.0

    if num > 0.0:
        ovx /= num
        
    return ovx, ch #, scoreV[0], indexV[0]

wf = {}
odir = './speechbrain_feature'
if not os.path.exists(odir):
    os.mkdir(odir)

for ct in ['cou', 'sub']:
    wf[ct] = open(f'{odir}/speechbrain_vec_{ct}.tsv', 'w')
    wf[ct].write(f'id\tstart\tend\tvector\temotion_hap-ang-sad-neu\n')

fdir='./fifty_split_wav'
for dirpath in glob.glob(f'{fdir}/*'):
    dir = dirpath.split('/')[-1]
    print(dir)
    id, ctype = dir.split('_')
    for wvpath in glob.glob(f'{dirpath}/*.wav'):
        print(wvpath)
        bn = os.path.basename(wvpath)
        bn = re.sub(r'\.wav$', '', bn)
        st, et = bn.split('_')
        ov, emo = estSpeechBrain(wvpath)
        estr = ''
        #for k,v in sorted(emo.items()):
        for e in ['hap', 'ang', 'sad', 'neu']:
            estr += f'{emo[e]:.3f} '
        estr = estr.rstrip(' ')
        vt = toStr(ov)
        wf[ctype].write(f'{id}\t{st}\t{et}\t{vt}\t{estr}\n')        



for k,v in wf.items():
    v.close

