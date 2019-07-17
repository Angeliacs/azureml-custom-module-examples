import os

import nltk
import numpy as np
import pyarrow.parquet as pq
import torch
from azureml.studio.common.io.data_table_io import read_data_table
from nltk.tokenize import word_tokenize
from torch.utils import data

nltk.download('punkt')


def process_data(args, file_name):
    """

    :return: word2id: map word to id
             id2word: map id to word
             label2id: map label to id
             id2label: map id to label
             max_len: max length of text
    """
    label2id, id2label, word2id, id2word, max_len = {}, {}, {}, {}, 0
    label_set, word_set = set([]), set([])

    parquet_path = os.path.join(file_name, 'data.dataset.parquet')
    dt = read_data_table(parquet_path)
    df = dt.data_frame

    for index, row in df.iterrows():
        label_set.add(row[args.label_column])
        sentence = row[args.text_column]
        words = word_tokenize(sentence)
        if len(words) > max_len:
            max_len = len(words)
        word_set |= set(words)

    id2word[0] = '<UNK>'  # unknown
    word2id['<UNK>'] = 0
    id2word[1] = '<EOS>'  # ending
    word2id['<EOS>'] = 1
    for i, word in enumerate(word_set):
        word2id[word] = i + 2
        id2word[i + 2] = word
    for i, label in enumerate(label_set):
        label2id[label] = i
        id2label[i] = label
    return word2id, id2word, label2id, id2label, max_len


def batch_collate(batch):
    idslist, label = zip(*batch)
    seq_lengths = torch.LongTensor([x.size for x in idslist])
    x = torch.ones((len(batch), seq_lengths.max())).long()
    for idx, (seq, seqlen) in enumerate(zip(idslist, seq_lengths)):
        if seq.size != 0:
            x[idx, :seqlen] = torch.from_numpy(seq)
    return x, torch.LongTensor(label)


def load_data(train_file, test_file, word2id, label2id, args):
    train_dataset = TextData(train_file, word2id, label2id, args, max_len=args.max_len)
    test_dataset = TextData(test_file, word2id, label2id, args, max_len=args.max_len)
    train_iter = data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        collate_fn=batch_collate,
        num_workers=8
    )
    test_iter = data.DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        collate_fn=batch_collate,
        num_workers=8
    )
    print(f"Load pyarrow.parquet explicitly: {pq}")
    return train_iter, test_iter


def sentence2idlist(sentence, word2id, max_len=-1):
    ids = [word2id[word] if word in word2id else 0 for word in word_tokenize(sentence)]
    if max_len > 0 and len(ids) > max_len:
        return ids[:max_len]
    return ids


class TextData(data.Dataset):
    def __init__(self, file, word2id, label2id, args, transform=sentence2idlist, max_len=-1):
        self.data = []
        self.transform = transform
        self.max_len = max_len
        parquet_path = os.path.join(file, 'data.dataset.parquet')
        dt = read_data_table(parquet_path)
        df = dt.data_frame

        for index, row in df.iterrows():
            self.data.append(
                (np.array(self.transform(row[args.text_column], word2id)), label2id[row[args.label_column]]))

        self.len = len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len
