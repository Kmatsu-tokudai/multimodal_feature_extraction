# Py-Feat特徴量を抽出する
#
# conda activate moviepy
#
import sys, os, re, glob
from feat import Detector


detector = Detector()

outdir = f'./pyfeat_feature_fifty'
if not os.path.exists(outdir):
    os.mkdir(outdir)

rdir = 'split_video_fifty'

# skip frames
num_frames = 15
if not os.path.exists(f'{outdir}'):
    os.mkdir(f'{outdir}')

for mdir in glob.glob(f'{rdir}/*'):
    id = mdir.split('/')[-1]
    if not os.path.exists(f'{outdir}/{id}'):
        os.mkdir(f'{outdir}/{id}')

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
        opath=f'{outdir}/{id}/{fn}.tsv'
        video_prediction.to_csv(opath, sep='\t')
        if num % 5 == 0:
            print( num, video_prediction.head(2))
        num += 1
