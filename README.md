# Session-aware BERT4Rec

Official repository for "Exploiting Session Information in BERT-based Session-aware Sequential Recommendation", SIGIR 2022 short.

Everything in the paper is implemented (including vanilla BERT4Rec and SASRec), and can be reproduced.

## Usage

### 1. Train

```bash
python train_onmo.py data_onmo
```
where data_onmo represents the folder containing the training data (onmo.csv)

### 2. Predict
```bash
python predict data_onmo data_onmo/onmo_predict.csv
```

where data_onmo represents the folder containing the pretrained model and data_onmo/onmo_predict.csv represents the file to predict from.
## Terminologies

The `df_` prefix always means DataFrame from Pandas.

* `uid` (str|int): User ID (unique).
* `iid` (str|int): Item ID (unique).
* `sid` (str|int): Session ID (unique), used only for session separation.
* `uindex` (int): mapped index number of User ID, 1 ~ n.
* `iindex` (int): mapped index number of Item ID, 1 ~ m.
* `timestamp` (int): UNIX timestamp.

## Data Files

After preprocessing, we'll have followings in each `data/:dataset_name/` directory.

* `uid2uindex.pkl` (dict): {`uid` &rightarrow; `uindex`}.
* `iid2iindex.pkl` (dict): {`iid` &rightarrow; `iindex`}.
* `df_rows.pkl` (df): column of (`uindex`, `iindex`, `sid`, `timestamp`), with no index.
* `train.pkl` (dict): {`uindex` &rightarrow; [list of (`iindex`, `sid`, `timestamp`)]}.
* `valid.pkl` (dict): {`uindex` &rightarrow; [list of (`iindex`, `sid`, `timestamp`)]}.
* `test.pkl` (dict): {`uindex` &rightarrow; [list of (`iindex`, `sid`, `timestamp`)]}.
* `ns_random.pkl` (dict): {`uindex` -> [list of `iindex`]}.
* `ns_popular.pkl` (dict): {`uindex` -> [list of `iindex`]}.

## Code References

* [FeiSun/BERT4Rec](https://github.com/FeiSun/BERT4Rec)
* [jaywonchung/BERT4Rec-VAE-Pytorch](https://github.com/jaywonchung/BERT4Rec-VAE-Pytorch)
