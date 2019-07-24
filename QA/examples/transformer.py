import json
import logging
import os

import pandas as pd
import pyarrow.parquet as pq

from args_util import process_args

logging.info(f"Load pyarrow.parquet explicitly: {pq}")


class DataPreprocessor(object):
    def process(self, data_frame: pd.DataFrame):
        context = data_frame['paragraphs']
        question = data_frame['question']
        title = data_frame['title']
        data = {'data': {'title': title, 'paragraphs': [{'context': context, 'qas': [{'question': question}]}]}}
        return data


if __name__ == '__main__':
    args = process_args()

    processor = DataPreprocessor()
    input_file_path = os.path.join(args.input_file, 'data.dataset.parquet')
    data_frame = pd.read_parquet(input_file_path, engine='pyarrow')
    data = processor.process(data_frame)

    json.dump(data, open(os.path.join(args.output_dir, 'dev.json'), "w"))
