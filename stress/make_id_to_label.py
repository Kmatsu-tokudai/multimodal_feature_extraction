# IDとラベルの対応
#
import sys, os, re, glob
import pandas as pd
import numpy as np

idir='./tsv'
wf = open('./label_data.tsv', 'w')
wf.write('ID\tscore_total\tscore_avg\tlabel\n')
scores, avgs = [], []
idh = {}
for path in glob.glob(f'{idir}/*.tsv'):
    bn = os.path.basename(path)
    id = re.sub(r'\.tsv$', '', bn)
    df = pd.read_csv(path, sep='\t')
    df = df[df['speaker'] == 'subject']
    sc, ag = 0.0, 0.0
    for score, avg in df[['総合得点_228（114-171）', '項目平均']].values:
        sc = score
        ag = avg
        break    
    idh[id] = [sc, ag]
    scores.append(sc)
    avgs.append(ag)


print(np.mean(avgs), np.max(avgs), np.min(avgs), np.median(avgs))
print(np.mean(scores), np.max(scores), np.min(scores), np.median(scores))

c = 0
for a in avgs:
    if a > np.median(avgs):
        c += 1

print(f"{c}/{len(avgs)}")


c = 0
high, low = [], []
for a in scores:
    if a > np.median(scores):
        high.append(a)        
        c += 1
    else:
        low.append(a)

print(f"{c}/{len(scores)}")

high.sort()
low.sort()
print(high)
print(low)


for id, v in idh.items():
    lb = 0
    if v[0] > np.median(scores):
        lb = 1
    
    wf.write(f'{id}\t{v[0]}\t{v[1]}\t{lb}\n')

wf.close
