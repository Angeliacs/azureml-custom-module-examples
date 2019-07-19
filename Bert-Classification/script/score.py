# coding=utf-8

from __future__ import absolute_import, division, print_function

import json
import logging
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
import torch
from azureml.core.run import Run
from pytorch_pretrained_bert.file_utils import WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.modeling import BertForSequenceClassification, BertConfig
from pytorch_pretrained_bert.tokenization import BertTokenizer
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import auc
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve
from sklearn.utils.fixes import signature
from torch.utils.data import (DataLoader, SequentialSampler,
                              TensorDataset)
from tqdm import tqdm

from .args_util import predict_args

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logging.info(f"Load pyarrow.parquet explicitly: {pq}")


class Classification:
    def __init__(self, model_dir: str):
        self.model, self.tokenizer, self.model_config = self.load_model(model_dir)
        self.label_map = self.model_config["label_map"]
        self.max_seq_length = self.model_config["max_seq_length"]
        self.label_map = {int(k): v for k, v in self.label_map.items()}
        self.model.eval()

    def load_model(self, model_dir, model_config: str = "model_config.json"):
        model_config = os.path.join(model_dir, model_config)
        model_config = json.load(open(model_config))
        output_config_file = os.path.join(model_dir, CONFIG_NAME)
        output_model_file = os.path.join(model_dir, WEIGHTS_NAME)
        config = BertConfig(output_config_file)
        model = BertForSequenceClassification(config, num_labels=model_config["num_labels"])
        model.load_state_dict(torch.load(output_model_file, map_location='cpu'))
        tokenizer = BertTokenizer.from_pretrained(model_config["bert_model"], do_lower_case=model_config["do_lower"])
        return model, tokenizer, model_config

    def predict(self, eval_dataloader, data_frame):
        model.eval()
        preds = []

        for input_ids, input_mask, segment_ids in tqdm(eval_dataloader, desc="Evaluating"):
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)

            with torch.no_grad():
                logits = model(input_ids, segment_ids, input_mask, labels=None)
            if len(preds) == 0:
                preds.append(logits.detach().cpu().numpy())
            else:
                preds[0] = np.append(preds[0], logits.detach().cpu().numpy(), axis=0)

        preds_prob = preds[0]
        preds = np.argmax(preds_prob, axis=1)
        data_frame['Scored Label'] = preds
        data_frame['Scored Prob'] = preds_prob
        print("preds", preds)

        if len(label_list) != 0:
            evaluation(all_label_ids.numpy(), preds, preds_prob, args.output_dir)
        return data_frame

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
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
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


def convert_examples_to_features(examples, max_seq_length, tokenizer, label_map):
    """Loads a data file into a list of `InputBatch`s."""

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        input_mask = [1] * len(input_ids)

        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        label_id = label_map[example.label]

        features.append(InputFeatures(
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids,
            label_id=label_id))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds)
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
    }


def pearson_and_spearman(preds, labels):
    pearson_corr = pearsonr(preds, labels)[0]
    spearman_corr = spearmanr(preds, labels)[0]
    return {
        "pearson": pearson_corr,
        "spearmanr": spearman_corr,
        "corr": (pearson_corr + spearman_corr) / 2,
    }


def compute_metrics(preds, labels):
    assert len(preds) == len(labels)
    return {"acc": simple_accuracy(preds, labels)}


def prcurve(df_true, df_predict, df_prob):
    average_precision = average_precision_score(df_true, df_predict)
    precision, recall, _ = precision_recall_curve(df_true, df_prob)
    step_kwargs = ({'step': 'post'}
                   if 'step' in signature(plt.fill_between).parameters
                   else {})
    f1_plt = plt.figure(1)
    plt.step(recall, precision, color='b', alpha=0.2,
             where='post')
    plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0, 1.1])
    plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(
        average_precision))

    return f1_plt


def scores(df_true, df_predict):
    f2_plt = plt.figure(2)
    metrics_name = ['precition', 'recall', 'F1-Score']
    p = precision_score(df_true, df_predict, average='binary')
    r = recall_score(df_true, df_predict, average='binary')
    f1 = f1_score(df_true, df_predict, average='binary')
    values_list = [p, r, f1]
    plt.bar(metrics_name, values_list, width=0.8, facecolor="#ff9999", edgecolor="white")

    for x, y in zip(metrics_name, values_list):
        plt.text(x, y, '%.4f' % y, ha='center', va='bottom')
    plt.ylim([0, 1.1])
    plt.ylabel('score')
    plt.title('Scores')

    return f2_plt


def Roc_curve(df_true, df_prob):
    f3_plt = plt.figure(3)
    fpr, tpr, thresholds = roc_curve(df_true, df_prob)
    roc_auc = auc(df_true, df_prob)

    plt.plot(fpr, tpr, label='ROC curve (area = %0.3f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate or (1 - Specifity)')
    plt.ylabel('True Positive Rate or (Sensitivity)')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")

    return f3_plt


def evaluation(df_true, df_predict, df_prob, output_eval_dir):
    run = Run.get_context()

    f1_plt = prcurve(df_true, df_predict, df_prob)
    run.log_image("precition/recall curve", plot=f1_plt)
    f1_plt.savefig(os.path.join(output_eval_dir, 'precition_recall.png'))

    f2_plt = scores(df_true, df_predict)
    run.log_image("scores", plot=f2_plt)
    f2_plt.savefig(os.path.join(output_eval_dir, 'scores.png'))

    f3_plt = Roc_curve(df_true, df_prob)
    run.log_image("ROC curve", plot=f3_plt)
    f3_plt.savefig(os.path.join(output_eval_dir, 'roc.png'))

def load_features(feature_dir, example_file: str = "examples", feature_config_file: str = "feature_config.json"):
    import pickle
    with open(os.path.join(feature_dir, example_file), "rb") as f:
        examples = pickle.load(f)
    text_list = []
    y_true = []
    for (ex_index, example) in enumerate(examples):
        text_list.append(example.text_a)
        try:
            y_true.append(example.label)
        except:
            continue

    feature_config = json.load(open(os.path.join(feature_dir, feature_config_file)))
    label_list = feature_config["label_list"]
    nums_labels = feature_config["num_labels"]

    return examples, label_list, nums_labels, text_list, y_true

if __name__ == "__main__":
    args = predict_args()

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    model_class = Classification(args.trained_model_dir)
    tokenizer = model_class.tokenizer
    model = model_class.model
    args.max_seq_length = model_class.max_seq_length

    model.to(device)
    eval_examples, label_list, num_labels, text_list, y_true = load_features(args.dev_file)
    eval_features = convert_examples_to_features(
        eval_examples, args.max_seq_length, tokenizer, model_class.label_map)

    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)

    eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids)
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)
    if len(y_true) == 0:
        data = {'text': text_list}
    else:
        data = {'text': text_list, 'label': y_true}

    import pandas as pd
    frame = pd.DataFrame(data)
    out_df = model_class.predict(eval_dataloader, frame)
    out_df.to_parquet(os.path.join(args.output_dir, 'data.dataset.parquet'))




