# Py-Feat特徴量を抽出する
#
# conda activate moviepy
#
import sys, os, re, glob
#import feat
from feat import Detector


detector = Detector()
    #face_model=img2pose,
    #landmark_model=mobilefacenet,
    #au_model=xgb,
    #emotion_model = resmasknet,
    #facepose_model = img2pose,
    #identity_model=facenet
    #data_type='video',

    #face_model="retinaface",
    ##landmark_model="mobilefacenet",
    ##au_model="xgb",
    ##emotion_model="resmasknet",
    #face_detection_threshold=0.95, 
#)

outdir = f'./pyfeat_feature_fifty'
if not os.path.exists(outdir):
    os.mkdir(outdir)

rdir = 'split_video_fifty'

num_frames = 15
#for aut in ['igakubu', 'riko']:
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
