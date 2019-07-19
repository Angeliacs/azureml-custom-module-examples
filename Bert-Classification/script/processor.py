# coding=utf-8
"""BERT finetuning runner."""

from __future__ import absolute_import, division, print_function

import json
import logging
import os
import pickle
import sys

import pandas as pd
import pyarrow.parquet as pq

from .args_util import process_args

logging.info(f"Load pyarrow.parquet explicitly: {pq}")


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_examples(self, train_file):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file):
        """Reads a tab separated value file."""
        df = pd.read_parquet(os.path.join(input_file, 'data.dataset.parquet'), engine='pyarrow')
        lines = []
        for index, row in df.iterrows():
            if sys.version_info[0] == 2:
                row = list(unicode(cell, 'utf-8') for cell in row)
            lines.append(row)
        return lines


class Processor(DataProcessor):
    """Processor for the SST-2 data set (GLUE version)."""

    def get_examples(self, file):
        """See base class."""
        # print(self._read_tsv(train_file))
        return self._create_examples(
            self._read_tsv(file))

    def _create_examples(self, lines):
        """Creates examples for the training and dev sets."""
        examples = []
        labels = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            text_a = line['text']
            try:
                label = line['label']
                examples.append(
                    InputExample(text_a=text_a, text_b=None, label=label))
                labels.append(label)
            except:
                examples.append(
                    InputExample(text_a=text_a, text_b=None))

        return examples, list(set(labels)), len(set(labels))


def main():
    args = process_args()
    processor = Processor()

    print("---------------tokenizer--------------------")
    examples, label_list, num_labels = processor.get_examples(args.input_file)
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
    with open(os.path.join(args.output_dir, "examples"), "wb") as f:
        pickle.dump(examples, f)
    features_data = {"label_list": label_list, "num_labels": num_labels}
    json.dump(features_data, open(os.path.join(args.output_dir, "feature_config.json"), "w"))


if __name__ == "__main__":
    main()
