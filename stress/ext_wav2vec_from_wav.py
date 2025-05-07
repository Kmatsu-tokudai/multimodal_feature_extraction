# ストレスデータ（音声）から，日本語wav2vec 2.0 (rinna)の特徴ベクトルを抽出する
import sys, os, glob, re
import soundfile as sf
from transformers import AutoFeatureExtractor, AutoModel
import numpy as np
import pandas as pd

def toStr(vec, delim):
    s = ''
    for v in vec:
        s += f'{v:3f}{delim}'
    s = s.rstrip(delim)
    return s

def get_vector( feat ):
    fm = []
    for f in feat[0]:
        fm.append(np.asarray(f))
    v = np.mean(fm, axis=0)
    return v #np.array(fm)

#model_name = "rinna/japanese-hubert-base"
model_name = "rinna/japanese-wav2vec2-base"
feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
model.eval()

rdir = './fifty_split_wav'

odir = './wav2vec'
if not os.path.exists(odir):
    os.mkdir(odir)

wf = open(f'{odir}/stress_fifty_wav2vec.tsv', 'w')
wf.write('ID\tstart_time\tend_time\twav2vec_vector\n')

for pdir in glob.glob(f'{rdir}/*'):
    for path in glob.glob(f'{pdir}/*.wav'):
        fid = path.split('/')[2]
        fnm = path.split('/')[3]
        fnm = re.sub(r'\.wav$', '', fnm)
        print(fid, fnm)
        stime, etime = fnm.split('_')

        audio_file = path #'./fifty_split_wav/2212182100_sub/1034450_1064450.wav'

        try:        
            raw_speech_16kHz, sr = sf.read(audio_file)
        except:
            print("sf read error!: ", path)
            continue

        try:
            inputs = feature_extractor(
                raw_speech_16kHz,    
                return_tensors="pt",
                sampling_rate=int(sr/2.0),
            )
            outputs = model(**inputs)
        except:
            print("error feature extractor: ", path)
            continue

        #print(f"Input:  {inputs.input_values.size()}")  # [1, #samples]
        #print(f"Output: {outputs.last_hidden_state.size()}")  # [1, #frames, 768]
        try:
            vec = get_vector(outputs.last_hidden_state.to('cpu').detach().numpy().copy())
        except:
            print("error get vector: ", path)
            continue
        delim = ' '
        vs = toStr(vec, delim)
        #print(vs)
        #va = vs.split(' ')
        #print(f"SIZE: {len(va)}")
        wf.write(f'{fid}\t{stime}\t{etime}\t{vs}\n')


wf.close
