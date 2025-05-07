# multimodal_feature_extraction
extract multimodal feature from audio, video, text data

- makeTextFeature.py
  - BERT(sentiment): nlptown/bert-base-multilingual-uncased-sentiment
  - E5(sentiment): Numind/e5-multilingual-sentiment_analysis
  - Sentiment_JA2:  https://github.com/sugiyamath/sentiment_ja2
  - RoSEtta: pkshatech/RoSEtta-base-ja

- extOpenSMILEfeature.py
  - FeatureSet.GeMAPSv01a
- extSpeechBrainFeature.py
  - https://huggingface.co/speechbrain/emotion-recognition-wav2vec2-IEMOCAP
- extPyFeatFeature.py
  - https://github.com/cosanlab/py-feat
- extHuBERTfeature.py
  - HuBERT: https://huggingface.co/rinna/japanese-hubert-base
  - Wav2Vec: https://huggingface.co/rinna/japanese-wav2vec2-base
  
- make_textFeature.py で使用するモデル(model.pkl)は，自分でダウンロードする

```
 with open("./model/model.pkl", "rb") as f:
    vect, models = pickle.load(f)
```

- 以下のSentiment_JA2という感情分析モデル
  - https://github.com/sugiyamath/sentiment_ja2


- 準備するコーパスは，音声データ，書き起こしテキスト，動画データを発話単位で分割したデータ
- 動画データはSimSwapで匿名化処理済み。フレームレートを10に設定してから匿名化。

- ストレスデータセット
  - カウンセラーとクライアントで分けて保存されている音声データ（発話単位で分割済み，ReazonSpeech nemo-asrで音声文字起こし済み）
  - この分割済みのデータをもとに動画ファイルも分割処理
  - ラベル付けは，閾値を中央値により定めている
  
- うつデータセット
  - 理工学部(riko)と医学部(igakubu)でディレクトリを分けている
  - そのほかは基本的に同じ
  - ラベル付けは，PHQ-9の中央値を基に定める（ストレス傾向を見たいので，うつ状態のレベルが高いか低いかがわかればよいため）
