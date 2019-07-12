import pyarrow.parquet as pq
import pickle
import re
import nltk
from nltk.tokenize import word_tokenize
import pyarrow.parquet as pq
from azureml.studio.common.io.data_table_io import read_data_table
from .args_util import *
from .TextCNN import *
import csv
import logging
import os

nltk.download('punkt')
class Predictor():
    def __init__(self, args): # model_path
        # model_args = xxx("snapshot/config.pkl", 'rb')
        self.args = args
        self.rule = re.compile(r"[^\u4e00-\u9fa5]")
        self.cut = word_tokenize
        with open(self.args.save_dir + '/' + 'word2id.pkl', 'rb') as f:
            self.word2id = pickle.load(f)
        with open(self.args.save_dir + '/' + 'id2label.pkl', 'rb') as f:
            self.id2label = pickle.load(f)

        with open(self.args.save_dir + '/' + 'config.pkl', 'rb') as f:
            config = pickle.load(f)
        config.predict_file = self.args.predict_file
        config.text_column = self.args.text_column
        config.save_dir = self.args.save_dir
        config.predict_result_file = self.args.predict_result_file
        config.logs = self.args.logs
        self.model = TextCNN(config)
        model_path = config.save_dir + "/best_steps_100.pt"
        if model_path is not None:
            self.model.load_state_dict(torch.load(model_path)) # snapshot = model path
        else:
            logging.warning('please input model path..')

        if self.args.cuda:
            torch.cuda.set_device(self.args.device)
            self.model = self.model.cuda()
        self.model.eval()

    def predict(self, text):
        # try:
        #     x = Variable(torch.LongTensor([[self.word2id[word] if word != '\x00' and word in self.word2id else 0 for word in text]]))
        #     if self.args.cuda:
        #         x = x.cuda()
        #     output = self.model(x)
        #
        #     _, predicted = torch.max(output, 1)
        #     predicted.numpy()
        #     return self.id2label[predicted.numpy()[0]]
        # except:
        #     return 'UNK'

        x = Variable(torch.LongTensor([[self.word2id[word] if word != '\x00' and word in self.word2id else 0 for word in text]]))
        if self.args.cuda:
            x = x.cuda()
        output = self.model(x)

        _, predicted = torch.max(output, 1)
        # predicted.numpy()
        return self.id2label[predicted.view(1).cpu().numpy()[0]]

if __name__ == '__main__':
    # with open("snapshot/config.pkl", 'rb') as f:
    #     args = pickle.load(f)
    args = get_args()
    if not os.path.isdir(args.logs):
        os.makedirs(args.logs)
    logname = args.logs + "/text.log"
    logging.basicConfig(filename=logname, filemode='w', level=logging.DEBUG)

    predictor = Predictor(args)
    if not os.path.isdir(args.predict_result_file):
        os.makedirs(args.predict_result_file)
    with open(args.predict_result_file+"/result.txt", "w", encoding="utf-8") as w:
        parquet_path = os.path.join(args.predict_file, 'data.dataset.parquet')
        dt = read_data_table(parquet_path)
        df = dt.data_frame
        for index, row in df.iterrows():
            sentence = row[args.text_column]
            words = word_tokenize(sentence)
            predictor.predict(words)
            w.write(str(predictor.predict(words)) + ',' + sentence + '\n')
