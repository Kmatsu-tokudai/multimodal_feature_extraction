# HuBERTによる音響特徴抽出
# conda activate speechbrain
#
import sys, os, re, glob
import pandas as pd
import numpy as np
import wave

# pydubは，音声ファイルのサンプリングレートの変換に用いる
from pydub import AudioSegment

# HuBERT/Wav2vec 2.0 用
import soundfile as sf
from transformers import AutoFeatureExtractor, AutoModel

w2v_model_name = "rinna/japanese-wav2vec2-base"
w2v_feature_extractor = AutoFeatureExtractor.from_pretrained(w2v_model_name)
w2vmodel = AutoModel.from_pretrained(w2v_model_name)
w2vmodel.eval()

hu_model_name = "rinna/japanese-hubert-base"
hu_feature_extractor = AutoFeatureExtractor.from_pretrained(hu_model_name)
hu_model = AutoModel.from_pretrained(hu_model_name)
hu_model.eval()

def chg_sampling_rate(wpath):   
    odir = './tmp_wav' 
    if not os.path.exists(odir):
        os.mkdir(odir)
    # 音声ファイルを読み込む
    audio = AudioSegment.from_file(wpath)
    # サンプリングレートを変更（例：16kHzに変更）
    audio_16k = audio.set_frame_rate(32000) 
    # 変更した音声を保存
    bn = os.path.basename(wpath)    
    wpath = re.sub(r'\.wav$', '', bn)
    new_wpath = f'{odir}/{wpath}_32k.wav'
    audio_16k.export(new_wpath, format="wav")
    return new_wpath

def toStr(vec):
    s = ''
    for v in vec:
        s += f'{v:.3f} '
    s = s.rstrip(' ')
    return s

def get_vector(self, feat):
    fm = []
    for f in feat[0]:
        fm.append(np.asarray(f))
    v = np.mean(fm, axis=0)
    return v

# HuBERT（音声特徴）の抽出
def extHuBERT(wpath):
    audio_file = wpath
    raw_speech_16kHz, sr = sf.read(audio_file)
    inputs = hu_feature_extractor(
        raw_speech_16kHz,    
        return_tensors="pt",
        sampling_rate=int(sr/2.0),
    )
    outputs = hu_model(**inputs)
    try:
        ovec = outputs.last_hidden_state.to('cpu').detach().numpy().copy()
        vec = np.mean(ovec[0], axis=0)         
    except Exception as e:
        print(e)
        print("hubert, Error: get_vector")
        
    return vec

# wav2vec 2.0 （音声特徴）の抽出
def extWav2Vec( wpath):

    raw_speech_16kHz, sr = sf.read(wpath)
    try:
        inputs = w2v_feature_extractor(
            raw_speech_16kHz,    
            return_tensors="pt",
            sampling_rate=int(sr/2.0),
        )
        outputs = w2vmodel(**inputs)
    except:
        print("error feature extractor: ", wpath)
        return "NULL"
    
    try:
        ovec = outputs.last_hidden_state.to('cpu').detach().numpy().copy()
        #print(ovec[0][0])
        vec = np.mean(ovec[0], axis=0) # get_vector(ovec[0][0])
    except:
        print("w2v error get vector: ", wpath)
        return "NULL"
    return vec


def toExt(speech_path, ftype):
    new_path = chg_sampling_rate(speech_path)
    with wave.open(new_path, 'rb') as wav_file:
        frame_rate = wav_file.getframerate()
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        frame_count = wav_file.getnframes()
        
        # 10秒ごとにファイルを区切る
        segment_length = 10 * frame_rate
    
        sumv = []
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

                # 特徴抽出
                try:
                    if ftype == 'hubert':
                        ov = extHuBERT(speech_path)
                    else:
                        ov = extWav2Vec(speech_path)
                except Exception as e:
                    print(e)
                    print(f"toExt {ftype} error!")
                    continue

                sumv.append(ov)

        if len(sumv) > 0:
            sumv = np.mean(sumv, axis=0)
        else:
            return []
        return sumv


for ftype in ['wav2vec', 'hubert']:
    print(f"Feature: {ftype} ...")
    odir = f'./{ftype}_feature_fifty'
    if not os.path.exists(odir):
        os.mkdir(odir)

    wf = {}
    for ct in ['cou', 'sub']:
        wf[ct] = open(f'{odir}/{ftype}_{ct}.tsv', 'w')
        wf[ct].write(f'id\tstart\tend\t{ftype}\n')
        
    fdir='./fifty_split_wav'
    for dirpath in glob.glob(f'{fdir}/*'):
        dir = dirpath.split('/')[-1]
        id, ctype = dir.split('_')
        for wvpath in glob.glob(f'{dirpath}/*.wav'):                
            bn = os.path.basename(wvpath)
            bn = re.sub(r'\.wav$', '', bn)
            st, et = bn.split('_')

            try:
                ov = toExt(wvpath, ftype)
                if len(ov) == 0:
                    print("ov zero: OV error!")
                    continue
            except Exception as e:
                print(e)
                print("Error!: ", wvpath)
                continue
            vt = toStr(ov)
            wf[ctype].write(f'{id}\t{st}\t{et}\t{vt}\n')        

    for k,v in wf.items():
        v.close

