import json
import logging
import os

import pandas as pd
import pyarrow.parquet as pq

from .args_util import process_args

logging.info(f"Load pyarrow.parquet explicitly: {pq}")


class DataPreprocessor(object):
    def process(self, data_frame: pd.DataFrame):
        data = {'data': []}
        for index, row in data_frame.iterrows():
            context = row['paragraphs']
            question = row['question']
            title = row['title']
            data['data'].append({'title': title, 'paragraphs': [{'context': context, 'qas': [{'question': question}]}]})
        return data


if __name__ == '__main__':
    args = process_args()

    processor = DataPreprocessor()
    input_file_path = os.path.join(args.input_file, 'data.dataset.parquet')
    data_frame = pd.read_parquet(input_file_path, engine='pyarrow')
    data = processor.process(data_frame)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    json.dump(data, open(os.path.join(args.output_dir, 'score.json'), "w"))
