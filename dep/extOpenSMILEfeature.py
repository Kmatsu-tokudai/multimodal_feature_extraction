# OpenSMILEによる音響特徴抽出
# conda activate opensmile
import sys, os, re, glob
import numpy as np
import opensmile

smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.GeMAPSv01a,
    feature_level=opensmile.FeatureLevel.Functionals,
)

odir = './opensmile_feature'
if not os.path.exists(odir):
    os.mkdir(odir)

fnames = smile.feature_names
of = open('./opensmile_feature/opensmile.GeMAPSv01a.tsv', 'w')
of.write('FeatureID\tFeatureName\n')
for i, f in enumerate(fnames):
    of.write(f'{i}\t{f}\n')
of.close

def toStr( fdf, fnames):
    fh = {}
    for i in range(len(fdf)):
        for f in fnames:
            if not f in fh:
                fh[f] = []
            fh[f].append(fdf.iloc[i][f])
    
    ss = ''
    for f in fnames:
        v = np.mean(fh[f], axis=0)
        ss += f'{v:.3f} '
    ss = ss.rstrip(' ')

    return ss

# opensmileで特徴抽出
def extOpenSMILE(speech_path):
    feature = smile.process_file(speech_path)
    return feature

wf = open(f'{odir}/dep_opensmile_vec.tsv', 'w')
wf.write(f'id\tstart\tend\tvector\n')

fdir='./split_wav'
for dirpath in glob.glob(f'{fdir}/*'):
    dir = dirpath.split('/')[-1]
    print(dir)
    id = dir 
    for wvpath in glob.glob(f'{dirpath}/*.wav'):
        bn = os.path.basename(wvpath)
        bn = re.sub(r'\.wav$', '', bn)
        st, et = bn.split('_')
        
        try:
            feature = extOpenSMILE(wvpath)
        except Exception as e:
            print(e)
            print("error: ", dir, st, et)
            continue

        vt = toStr(feature, fnames)
        wf.write(f'{id}\t{st}\t{et}\t{vt}\n')

wf.close


