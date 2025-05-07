# mp4ファイルを，発話区間単位で分割する
#
# conda activate moviepy
# 
import ffmpeg
import sys, os, re, glob
import numpy as np
import pandas as pd

import os
import glob
import sys



# メイン処理
if __name__ == "__main__":
    rdir = 'video_and_audio/video'
    odir = 'split_video'
    if not os.path.exists(odir):
        os.mkdir(odir)
    
    for at in ['igakubu', 'riko']:
        dir = f'{odir}/{at}'
        if not os.path.exists(dir):
            os.mkdir(dir)

    df = pd.read_csv('output_transcription.tsv', sep='\t')
    for i in range(len(df)):
        at = df.iloc[i]['AuthorType']
        fn = df.iloc[i]['FileName']        
        tfn = re.sub(r'\.m4a$', '', fn)
        fn = re.sub(r'\.m4a$', '.mp4', fn)
        
        todir = f'{odir}/{at}/{tfn}'
        if not os.path.exists(todir):
            os.mkdir(todir)

        st = df.iloc[i]['Start (ms)']
        en = df.iloc[i]['End (ms)']
        ofn = f'{st}_{en}.mp4'

        fpath = f'{rdir}/{at}/{fn}'
        if os.path.exists(fpath):
            opath = f'{todir}/{ofn}'
            print(fpath, "\t", st, "====", en)

            span = en - st
            span = float(span) / 1000.0
            st = float(st) / 1000.0
            process = (
                ffmpeg
                .input(fpath,ss=st, t=span)
                .filter('fps', fps=10, round='up')
                .output(opath, crf=10)
                .run_async(pipe_stdin=True, cmd='/usr/bin/ffmpeg')
            )
            process.stdin.close()
            process.wait()