# ReazonSpeechで音声認識
# 無音区間で音声データを分割
# conda activate reazon
import sys, os, glob
import re
from reazonspeech.nemo.asr import load_model, transcribe, audio_from_path
import yaml
import pathlib
import numpy as np

# 無音区間検出器
from inaSpeechSegmenter import Segmenter

import torch
from pydub import AudioSegment
import wave


print("load...")
model = load_model()#
print("load ok!")

def toStr(vec):
    ss = ''
    for m in vec:
        ss += '{:.5f} '.format(m)
    ss = ss.rstrip(' ')
    return ss

import os

os.environ['HUGGINGFACE_HUB_CACHE'] = './assets'

# HuggingFaceHubのトークンは各自で取得する
HF_TOKEN='TOKEN'

# 1. 音声ファイルの読み込み
sounddir='./video_and_audio/audio'
sounds = []
metadata = []
for ctype in ['riko', 'igakubu']:
    print(ctype)
    for path in glob.glob(f'{sounddir}/{ctype}/*'):
        audio_file = path #"your_audio_file.m4a"
        bn = os.path.basename(audio_file)
        sounds.append(audio_file)
        metadata.append([ctype, bn])

print("load file ok.")

# 3. 音声認識とTSVファイルの出力
def transcribe_audio(sound, wf, meta, output_dir):
    seg = Segmenter(vad_engine='smn', detect_gender=False)
    segmentation = seg(sound)
    checks = {}
    nAudio, audio_chunk = '', ''
    for segment in segmentation:
        segment_label = segment[0]
        if segment_label == 'speech':
            start = int(segment[1] * 1000)
            end = int(segment[2] * 1000)
            if start in checks:
                print(f"already check!: {start}--{end}")
                continue
            checks[start] = 1
            nAudio = AudioSegment.from_file(sound)            
            audio_chunk = nAudio[start:end]
            outfile = f'{output_dir}/{start}_{end}.wav'
            audio_chunk.export(outfile, format='wav')
            aud = audio_from_path(outfile)
            ret = transcribe(model, aud)

            # ReazonSpeech を使った文字起こし(transcribe)
            try:
                tras = ret.text
                wf.write(f"{meta[0]}\t{meta[1]}\t{start}\t{end}\t{tras}\n")
                del nAudio
                del audio_chunk
            except Exception as e:
                print("Error!: ", e)
                continue
                #wf.write(f"{meta[0]}\t{meta[1]}\t{start}\t{end}\tError: {response.status_code}\n")
                del audio_chunk
                del nAudio

# 実行
output_tsv = "output_transcription.tsv"
wf = open(f'{output_tsv}', 'w')
wf.write("AuthorType\tFileName\tStart (ms)\tEnd (ms)\tTranscription\n")
outdir = './split_wav'
if not os.path.exists(outdir):
    os.mkdir(outdir)

for sound, meta in zip(sounds, metadata):
    dn = sound.split('/')[-1]
    dn = re.sub(r'\.m4a$', '', dn)
    print("==>", dn)
    output_dir = f'{outdir}/{dn}'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    transcribe_audio(sound, wf, meta, output_dir)
wf.close
