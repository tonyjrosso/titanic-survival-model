# 機械学習モデル更新時の性能検証

以下の機能を備えた、クリーンで本番環境レベルの機械学習プロジェクトです。

- モジュール型のプロジェクトアーキテクチャ
- 再現可能なトレーニングパイプライン
- モデルレジストリ
- メトリクスの計算
- 可視化（グラウンドトゥルースの有無にかかわらず）
- クリーンなモデル再学習ワークフロー
- データセットの読み込みと前処理

**重要**: この[ノートブック](https://github.com/tonyjrosso/titanic-survival-model/blob/main/titanic_survival_model/notebooks/Titanic%20Survival%20Model%20demo.ipynb)には、プロジェクトに関する詳細な議論が記載されています。

このプロジェクトでは、タイタニック号のデータセットを使用して、乗客の生存を予測するロジスティック回帰分類器を構築します。

## プロジェクト構造

```
titanic_survival_model/
│
├── model/
│   ├── inference.py          # predict() and predict_with_report()
│   ├── training.py           # train() with CV, test split, metrics, saving
│   ├── registry.py           # load/save model artifacts + metadata
│   └── artifacts/            # serialized models + metadata files
│
├── metrics/
│   ├── evaluation.py         # compute_metrics() producing structured metrics
│   ├── monitoring.py         # collects performance history over model versions
│   ├── visualization.py      # score/ROC/PR/confusion plots
│   └── schema.py             # schemas for metrics
│
├── data/
│   └── loader.py             # loads + preprocesses Titanic dataset
│
├── config/
│   ├── settings.yaml         # default training + path configs
│   └── load_config.py        # loads configs for other modules
│
├── notebooks/
│   └── Titanic_Demo.ipynb    # walkthrough for training + evaluation
│
└── requirements.txt
```


## 実行方法
1. 依存関係をインストールする
```
pip install -r requirements.txt
```

2. デモノートブックを実行する
```
jupyter notebook notebooks/Titanic_Demo.ipynb
```

3. あるいは新しいモデルをプログラムでトレーニングする
```
from model.training import train
from data.loader import load_titanic
from sklearn.linear_model import LogisticRegression

X, y = load_titanic()
model, metrics = train(LogisticRegression(), X, y)
```

## 構成

`config/settings.yaml`を編集して、以下の設定を変更します。
- トレーニング/テストの分割
- ランダムシード
- アーティファクトディレクトリ


## モデルアーティファクト

トレーニングされたモデルは `model/artifacts/` に保存されます。

## 謝辞

データセットは以下のソースから取得されました。
> https://github.com/datasciencedojo/datasets/blob/master/titanic.csv
