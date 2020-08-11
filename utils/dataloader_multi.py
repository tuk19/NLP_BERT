import glob
import os
import io
import string
import re
import sys
import random
import spacy
import torchtext
import mojimoji
import string
import time
import numpy as np
from tqdm import tqdm
import torch 
from torch import nn
import torch.optim as optim
import torchtext
import pickle
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from utils.text_id import JumanppBERTTextProcessor, BertWordPieceTokenizer
from pyknp import Juman

from torchtext.vocab import Vectors
from utils.bert import BertTokenizer, load_vocab
from utils.config import PKL_FILE, VOCAB_FILE, DATA_PATH


class CreateDataset(Dataset):
  def __init__(self, X, y, tokenizer, max_len):
    self.X = X
    self.y = y
    self.tokenizer = tokenizer
    self.max_len = max_len

  def __len__(self):  # len(Dataset)で返す値を指定
    return len(self.y)

  def __getitem__(self, index):  # Dataset[index]で返す値を指定
    text = self.X[index]
    # inputs = self.tokenizer(text)
    jpp = Juman()
    bertwordpiecetokenizer = BertWordPieceTokenizer()
    jbp = JumanppBERTTextProcessor(jpp, bertwordpiecetokenizer, fix_length=256)
    ids = jbp.to_bert_input(text)

    return {
      'ids': torch.LongTensor(ids),
      'labels': torch.Tensor(self.y[index])
    }


  def test(self, index):  # Dataset[index]で返す値を指定
    text = self.X[index]
    inputs = self.tokenizer(text)
    ids = inputs
    print(ids)


"""
class MultiLabelDataSet(torch.utils.data.Dataset):
    def __init__(self, labels, text_file='/content/drive/My Drive/Colab Notebooks/my_bert/data/test_bin.tsv',　ext='.jpg', transform=None):
        self.labels = labels
        self.text_file = text_file
        self.ext = ext
        self.transform = transform

        self.keys = list(labels.keys())
        self.vals = list(labels.values())

    def __len__(self):
        return len(self.labels)

    def preprocessing_text(text):
        # 半角・全角の統一
        text = mojimoji.han_to_zen(text) 
        # 改行、半角スペース、全角スペースを削除
        text = re.sub('\r', '', text)
        text = re.sub('\n', '', text)
        text = re.sub('　', '', text)
        text = re.sub(' ', '', text)
        # 数字文字の一律「0」化
        text = re.sub(r'[0-9 ０-９]+', '0', text)  # 数字

        # カンマ、ピリオド以外の記号をスペースに置換
        for p in string.punctuation:
            if (p == ".") or (p == ","):
                continue
            else:
                text = text.replace(p, " ")
            return text
    
    # 前処理と単語分割をまとめた関数を定義
    # 単語分割の関数を渡すので、tokenizer_bertではなく、tokenizer_bert.tokenizeを渡す点に注意
    def tokenizer_with_preprocessing(text, tokenizer=tokenizer_bert.tokenize):
        text = preprocessing_text(text)
        ret = tokenizer(text)  # tokenizer_bert
        return ret

    def __getitem__(self, idx):
        tokenizer_bert = BertTokenizer(vocab_file="/content/drive/My Drive/Colab Notebooks/bert_最終課題/vocab/vocab.txt", do_lower_case=False)
        df = pd.read_csv(text_file, delimiter="\t")
        
        TEXT = torchtext.data.Field(sequential=True, tokenize=tokenizer_with_preprocessing, use_vocab=True,
                            lower=False, include_lengths=True, batch_first=True, fix_length=max_length, init_token="[CLS]", eos_token="[SEP]", pad_token='[PAD]', unk_token='[UNK]')
        
        


        LABEL = torch.utils.data.TensorDataset(torch.from_numpy(trX).float(), torch.from_numpy(trY.astype(np.int64)))

        

        text_path = f'{self.text_dir}/{self.keys[idx]}{self.ext}'
        text_array = Image.open(text_path)
        if self.transform:
            text = self.transform(image_array)
        else:
            text = torch.Tensor(np.transpose(image_array, (2, 0, 1)))/255  # for 0~1 scaling

        label = torch.Tensor(list(self.vals[idx].values()))

        return {'Text': TEXT, 'Label': LABEL}
"""
def get_chABSA_DataLoaders_and_TEXT(max_length=256, batch_size=32):
    """IMDbのDataLoaderとTEXTオブジェクトを取得する。 """
    # 乱数のシードを設定
    torch.manual_seed(1234)
    np.random.seed(1234)
    random.seed(1234)
    # 単語分割用のTokenizerを用意
    tokenizer_bert = BertTokenizer(vocab_file="/content/drive/My Drive/Colab Notebooks/bert_最終課題/vocab/vocab.txt", do_lower_case=False)

    def preprocessing_text(text):
        # 半角・全角の統一
        text = mojimoji.han_to_zen(text) 
        # 改行、半角スペース、全角スペースを削除
        text = re.sub('\r', '', text)
        text = re.sub('\n', '', text)
        text = re.sub('　', '', text)
        text = re.sub(' ', '', text)
        # 数字文字の一律「0」化
        text = re.sub(r'[0-9 ０-９]+', '0', text)  # 数字

        # カンマ、ピリオド以外の記号をスペースに置換
        for p in string.punctuation:
            if (p == ".") or (p == ","):
                continue
            else:
                text = text.replace(p, " ")
            return text

    # 前処理と単語分割をまとめた関数を定義
    # 単語分割の関数を渡すので、tokenizer_bertではなく、tokenizer_bert.tokenizeを渡す点に注意
    def tokenizer_with_preprocessing(text, tokenizer=tokenizer_bert.tokenize):
        text = preprocessing_text(text)
        ret = tokenizer(text)  # tokenizer_bert
        return ret
    # データを読み込んだときに、読み込んだ内容に対して行う処理を定義します
    max_length = 256
    TEXT = torchtext.data.Field(sequential=True, tokenize=tokenizer_with_preprocessing, use_vocab=True,
                            lower=False, include_lengths=True, batch_first=True, fix_length=max_length, init_token="[CLS]", eos_token="[SEP]", pad_token='[PAD]', unk_token='[UNK]')
    LABEL = torchtext.data.Field(sequential=False, use_vocab=False)

    # フォルダ「data」から各tsvファイルを読み込みます
    # BERT用で処理するので、10分弱時間がかかります
    train_val_ds, test_ds = torchtext.data.TabularDataset.splits(
        path=DATA_PATH, train='train.tsv',
        test='test.tsv', format='tsv',
        fields=[('Text', TEXT), ('Label', LABEL)])

    vocab_bert, ids_to_tokens_bert = load_vocab(vocab_file=VOCAB_FILE)
    TEXT.build_vocab(train_val_ds, min_freq=1)
    TEXT.vocab.stoi = vocab_bert    
    
    batch_size = 32  # BERTでは16、32あたりを使用する
    train_dl = torchtext.data.Iterator(train_val_ds, batch_size=batch_size, train=True)
    val_dl = torchtext.data.Iterator(test_ds, batch_size=batch_size, train=False, sort=False)
    # 辞書オブジェクトにまとめる
    dataloaders_dict = {"train": train_dl, "val": val_dl}

    return train_dl, val_dl, TEXT, dataloaders_dict

