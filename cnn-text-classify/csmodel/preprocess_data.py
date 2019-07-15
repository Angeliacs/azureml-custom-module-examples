import pyarrow.parquet as pq

import os
import pickle
import re

import pandas as pd
from nltk.tokenize import word_tokenize
import nltk
from .args_util import preprocess_args

nltk.download('punkt')
class DataPreprocessor(object):
    def __init__(self, vocab_path):
        self.vocab_path = vocab_path
        self.rule = re.compile(r"[^\u4e00-\u9fa5]")
        self.cut = word_tokenize
        with open(self.vocab_path + '/' + 'word2id.pkl', 'rb') as f:
            self.word2id = pickle.load(f)

    def process(self, data_frame: pd.DataFrame):
        word_id_list = []
        label_list = []
        for index, row in data_frame.iterrows():
            text = word_tokenize(row['text'])
            text_id = [self.word2id[word] if word != '\x00' and word in self.word2id else 0 for word in text]
            word_id_list.append(text_id)
            label_list.append(row['label'])
        return pd.DataFrame({'label': label_list, 'text_id': word_id_list})


if __name__ == '__main__':
    args = preprocess_args()
    processor = DataPreprocessor(args.input_vocab)
    input_file_path = os.path.join(args.input_data, 'data.dataset.parquet')
    data_frame = pd.read_parquet(input_file_path, engine='pyarrow')
    output_data = processor.process(data_frame)
    if not os.path.exists(args.output_data):
        os.makedirs(args.output_data)
    print("write to ", os.path.join(args.output_data, 'data.dataset.parquet'))
    output_data.to_parquet(os.path.join(args.output_data, 'data.dataset.parquet'), engine='pyarrow')

    import json

    # Dump data_type.json as a work around until SMT deploys
    dct = {
        "Id": "Dataset",
        "Name": "Dataset .NET file",
        "ShortName": "Dataset",
        "Description": "A serialized DataTable supporting partial reads and writes",
        "IsDirectory": False,
        "Owner": "Microsoft Corporation",
        "FileExtension": "dataset.parquet",
        "ContentType": "application/octet-stream",
        "AllowUpload": False,
        "AllowPromotion": True,
        "AllowModelPromotion": False,
        "AuxiliaryFileExtension": None,
        "AuxiliaryContentType": None
    }
    with open(os.path.join(args.output_data, 'data_type.json'), 'w') as f:
        json.dump(dct, f)

