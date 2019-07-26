""" Finetuning the library models for question-answering on SQuAD (Bert, XLM, XLNet)."""

from __future__ import absolute_import, division, print_function

# The follwing import is the official SQuAD evaluation script (2.0).
# You can remove it from the dependencies if you are using this script outside of the library
# We've added it here for automated tests (see examples/test_examples.py file)
import json
import logging
import os
import random

import numpy as np
import torch
from pytorch_transformers import (BertConfig,
                                  BertForQuestionAnswering, BertTokenizer,
                                  XLMConfig, XLMForQuestionAnswering,
                                  XLMTokenizer, XLNetConfig,
                                  XLNetForQuestionAnswering,
                                  XLNetTokenizer)
from torch.utils.data import (DataLoader, SequentialSampler,
                              TensorDataset)
from tqdm import tqdm

from .args_util import predict_args
from .utils_squad import (convert_examples_to_features,
                         RawResult, write_predictions,
                         RawResultExtended, write_predictions_extended)

logger = logging.getLogger(__name__)

ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) \
                  for conf in (BertConfig, XLNetConfig, XLMConfig)), ())

MODEL_CLASSES = {
    'bert': (BertConfig, BertForQuestionAnswering, BertTokenizer),
    'xlnet': (XLNetConfig, XLNetForQuestionAnswering, XLNetTokenizer),
    'xlm': (XLMConfig, XLMForQuestionAnswering, XLMTokenizer),
}


class SquadExample(object):
    """
    A single training/test example for the Squad dataset.
    For examples without an answer, the start and end position are -1.
    """

    def __init__(self,
                 question_text,
                 doc_tokens,
                 qas_id=None,
                 orig_answer_text=None,
                 start_position=None,
                 end_position=None,
                 is_impossible=None):
        self.qas_id = qas_id
        self.question_text = question_text
        self.doc_tokens = doc_tokens
        self.orig_answer_text = orig_answer_text
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "qas_id: %s" % (self.qas_id)
        s += ", question_text: %s" % (
            self.question_text)
        s += ", doc_tokens: [%s]" % (" ".join(self.doc_tokens))
        if self.start_position:
            s += ", start_position: %d" % (self.start_position)
        if self.end_position:
            s += ", end_position: %d" % (self.end_position)
        if self.is_impossible:
            s += ", is_impossible: %r" % (self.is_impossible)
        return s


class QA:
    # todo : support batch input
    def __init__(self, model_dir: str):
        self.model, self.tokenizer, self.model_config = self.load_model(model_dir)
        self.max_seq_length = self.model_config["max_seq_length"]
        self.max_query_length = self.model_config["max_query_length"]
        self.max_answer_length = self.model_config["max_answer_length"]
        self.do_lower_case = self.model_config["do_lower"]
        self.model.eval()

    def load_model(self, model_dir, model_config: str = "model_config.json"):
        model_config = os.path.join(model_dir, model_config)
        model_config = json.load(open(model_config))
        config_class, model_class, tokenizer_class = MODEL_CLASSES[model_config['model_type']]
        model = model_class.from_pretrained(model_dir)
        tokenizer = tokenizer_class.from_pretrained(model_dir)
        return model, tokenizer, model_config

    def predict(self, input_data):
        # Setup CUDA, GPU & distributed training
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        def is_whitespace(c):
            if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
                return True
            return False

        self.model.to(device)
        examples = []
        for entry in input_data['data']:
            for paragraph in entry["paragraphs"]:
                paragraph_text = paragraph["context"]
                doc_tokens = []
                char_to_word_offset = []
                prev_is_whitespace = True
                for c in paragraph_text:
                    if is_whitespace(c):
                        prev_is_whitespace = True
                    else:
                        if prev_is_whitespace:
                            doc_tokens.append(c)
                        else:
                            doc_tokens[-1] += c
                        prev_is_whitespace = False
                    char_to_word_offset.append(len(doc_tokens) - 1)

                for qa in paragraph["qas"]:
                    try:
                        qas_id = qa["id"]
                    except:
                        qas_id = None
                    question_text = qa["question"]
                    start_position = None
                    end_position = None
                    orig_answer_text = None
                    is_impossible = False
                    example = SquadExample(
                        qas_id=qas_id,
                        question_text=question_text,
                        doc_tokens=doc_tokens,
                        orig_answer_text=orig_answer_text,
                        start_position=start_position,
                        end_position=end_position,
                        is_impossible=is_impossible)
                    examples.append(example)
        dataset, examples, features = load_and_cache_examples(examples, self.max_seq_length, self.max_query_length,
                                                              self.tokenizer)

        all_results = evaluate(device, self.model_config, dataset, features, self.model)

        import pandas as pd
        predictions = pd.DataFrame()
        if self.model_config['model_type'] in ['xlnet', 'xlm']:
            # XLNet uses a more complex post-processing procedure
            predictions = write_predictions_extended(examples, features, all_results, 1,
                                                     self.max_answer_length, input_data,
                                                     self.model.config.start_n_top, self.model.config.end_n_top,
                                                     self.tokenizer)

        else:
            predictions = write_predictions(examples, features, all_results, 1,
                                            self.max_answer_length, self.do_lower_case)

        return predictions


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def to_list(tensor):
    return tensor.detach().cpu().tolist()


def evaluate(device, model_config, dataset, features, model):
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=1)

    # Eval!
    all_results = []
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            inputs = {'input_ids': batch[0],
                      'token_type_ids': None if model_config['model_type'] == 'xlm' else batch[1],
                      # XLM don't use segment_ids
                      'attention_mask': batch[2]}
            example_indices = batch[3]
            if model_config['model_type'] in ['xlnet', 'xlm']:
                inputs.update({'cls_index': batch[4],
                               'p_mask': batch[5]})
            outputs = model(**inputs)

        for i, example_index in enumerate(example_indices):
            eval_feature = features[example_index.item()]
            unique_id = int(eval_feature.unique_id)
            if model_config['model_type'] in ['xlnet', 'xlm']:
                # XLNet uses a more complex post-processing procedure
                result = RawResultExtended(unique_id=unique_id,
                                           start_top_log_probs=to_list(outputs[0][i]),
                                           start_top_index=to_list(outputs[1][i]),
                                           end_top_log_probs=to_list(outputs[2][i]),
                                           end_top_index=to_list(outputs[3][i]),
                                           cls_logits=to_list(outputs[4][i]))
            else:
                result = RawResult(unique_id=unique_id,
                                   start_logits=to_list(outputs[0][i]),
                                   end_logits=to_list(outputs[1][i]))
            all_results.append(result)

    return all_results


def load_and_cache_examples(examples, max_seq_length, max_query_length, tokenizer):
    features = convert_examples_to_features(examples=examples,
                                            tokenizer=tokenizer,
                                            max_seq_length=max_seq_length,
                                            doc_stride=128,
                                            max_query_length=max_query_length,
                                            is_training=not evaluate)

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_cls_index = torch.tensor([f.cls_index for f in features], dtype=torch.long)
    all_p_mask = torch.tensor([f.p_mask for f in features], dtype=torch.float)

    all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                            all_example_index, all_cls_index, all_p_mask)

    return dataset, examples, features


def main():
    args = predict_args()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    predictor = QA(args.trained_model)
    with open(args.input_file, "r", encoding='utf-8') as reader:
        input_data = json.load(reader)
    all_results = predictor.predict(input_data)
    all_results.to_parquet(os.path.join(args.output_dir, 'data.dataset.parquet'))


if __name__ == "__main__":
    main()
