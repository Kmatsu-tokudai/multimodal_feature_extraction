# BERT-senti, E5-senti, RoSEtta, sentimentjA2
# などの特徴量抽出を行う
# conda activate bert

import sys, os, re, glob
import numpy as np
import pandas as pd

# Sentiment_Ja2用 (モデルファイル必要）
import pickle
import pprint
from sudachipy import tokenizer
from sudachipy import dictionary

# pkshatech/RoSEtta-base-ja(テキスト特徴量)用
from sentence_transformers import SentenceTransformer

# E5-large-japanese用
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
# BERT('nlptown/bert-base-multilingual-uncased-sentiment')も使用
import torch

def toStr(vec):
    s = ''
    for v in vec:
        s += f'{v:.3f} '
    s = s.rstrip(' ')
    return s

rosetta_model = SentenceTransformer("pkshatech/RoSEtta-base-ja",trust_remote_code=True)
e5_tokenizer = AutoTokenizer.from_pretrained("Numind/e5-multilingual-sentiment_analysis")
e5_model = AutoModel.from_pretrained("Numind/e5-multilingual-sentiment_analysis")
bert_modelname='nlptown/bert-base-multilingual-uncased-sentiment'
bert_model = AutoModel.from_pretrained(bert_modelname)
bert_tokenizer = AutoTokenizer.from_pretrained(bert_modelname)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
bert_model.to(device)

with open("./model/model.pkl", "rb") as f:
    vect, models = pickle.load(f)

tokenizer_obj = dictionary.Dictionary().create()
mode = tokenizer.Tokenizer.SplitMode.C

# 平均プーリング
def average_pool(last_hidden_states: Tensor,
                attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

# Rosetta（テキストベクトル）の抽出
def extRosetta( input_texts):
    #model = SentenceTransformer("pkshatech/RoSEtta-base-ja",trust_remote_code=True)
    embeddings = rosetta_model.encode( input_texts, convert_to_tensor=True)
    return embeddings

# E5（テキストベクトル）の抽出
def extE5( input_texts):
    # Tokenize the input texts
    batch_dict = e5_tokenizer(input_texts, max_length=512, padding=True, truncation=True, return_tensors='pt')
    outputs = e5_model(**batch_dict)
    embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
    embeddings = F.normalize(embeddings, p=2, dim=1)
    scores = (embeddings[:2] @ embeddings[2:].T) * 100  
    emb = embeddings.to('cpu').detach().numpy().copy()
    #emb = torch.mean(emb, axis = 1)
    #emb = np.reshape(emb, (768))      
    return emb #eddings #scores

# BERT(sentiment)の抽出
def extBERTsenti( input_texts):
    size = 256
    text = input_texts[0]
    encoding = bert_tokenizer(
        text,
        truncation=True, 
        padding='max_length', 
        max_length= size,
    )

    emb = bert_model(
        torch.reshape(torch.tensor(encoding.input_ids),(1,len(encoding.input_ids))).to(device),output_hidden_states=True
    ).hidden_states[-1].cpu().detach()

    print("Emb length:" ,len(emb[0]))
    emb = torch.mean(emb[0], axis = 0) #1)
    emb = np.reshape(emb, (768))
    return emb

def tok( x, tokenizer_obj, mode):
    return ' '.join([m.dictionary_form() for m in tokenizer_obj.tokenize(x, mode)]).strip()

# SentimentJA2によるテキスト感情分析
def extSJA2( input_texts):
    emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise']
    v = vect.transform(tok(x, tokenizer_obj, mode) for x in input_texts)
    out = [{"text": x} for x in input_texts]
    for n, m in models:
        for i, p in enumerate(m.predict_proba(v)[:,1]):
            out[i][n] = p

    results = []
    for i, (txt, o) in enumerate(zip(input_texts, out)):
        vv = []
        for e in emotions:
            vv.append(o[e])
        results.append([txt, vv])

    return results

def get_vector( feat):
    fm = []
    for f in feat[0]:
        fm.append(np.asarray(f))
    v = np.mean(fm, axis=0)
    return v


def main():
    wf = open('./dep_data/dep_text_feature.tsv', 'w')
    wf.write('authortype\tfilename\tstart\tend\ttext\tBERT\tE5\tSJ2\tRosetta\n')
    df = pd.read_csv('./dep_data/output_transcription.tsv', sep='\t')
    for i in range(len(df)):
        st = df.iloc[i]['Start (ms)']
        en = df.iloc[i]['End (ms)']
        fn = df.iloc[i]['FileName']
        at = df.iloc[i]['AuthorType']
        txt = df.iloc[i]['Transcription']
        
        print("Text: ", txt)
        if f"{txt}" == 'nan':
            continue
            
        try:
            bert = extBERTsenti([txt])
        except Exception as e:            
            print("Bert Error:", txt)
            break

        try:
            e5 = extE5([txt])
        except Exception as e:
            print("E5 error: ", txt)
            break

        try:
            rose = extRosetta([txt])
        except Exception as e:
            print("Rosetta error:", txt)
            break
        
        try:
            sj2 = extSJA2([txt])
        except Exception as e:
            print("SentiJA2 error!", txt)
            break

        bstr = toStr(bert)
        estr = toStr(e5[0])

        vec = []
        for sd in sj2:
            vec.append(sd[1])
        vec = np.mean(vec, axis=0)
        sjs = toStr(vec)
        rss = toStr(rose[0])
        wf.write(f'{at}\t{fn}\t{st}\t{en}\t{txt}\t{bstr}\t{estr}\t{sjs}\t{rss}\n')
    
    wf.close


if __name__ == '__main__':
    main()
