# Py-Feat特徴量を抽出する
#
# conda activate moviepy
#
import sys, os, re, glob
from feat import Detector


detector = Detector()

outdir = f'./pyfeat_feature'
if not os.path.exists(outdir):
    os.mkdir(outdir)

rdir = 'split_video'

num_frames = 15
for aut in ['igakubu', 'riko']:
    if not os.path.exists(f'{outdir}/{aut}'):
        os.mkdir(f'{outdir}/{aut}')

    for mdir in glob.glob(f'{rdir}/{aut}/*'):
        id = mdir.split('/')[-1]
        if not os.path.exists(f'{outdir}/{aut}/{id}'):
            os.mkdir(f'{outdir}/{aut}/{id}')

        num = 0
        for test_video_path in glob.glob(f'{mdir}/*.mp4'):
            try:
                video_prediction = detector.detect(test_video_path, data_type='video', skip_frames=num_frames, 
                face_detection_threshold=0.95)
            except Exception as e:
                print(e)
                print("Error! id: ", id, test_video_path)
                continue
            fn = os.path.basename(test_video_path)
            fn = re.sub(r'\.mp4$', '', fn)
            opath=f'{outdir}/{aut}/{id}/{fn}.tsv'
            video_prediction.to_csv(opath, sep='\t')
            num += 1
