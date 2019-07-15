import pyarrow.parquet as pq
import os
import pickle

import pandas as pd
import torch
from .TextCNN import TextCNN
from .args_util import predict_args
from torch.autograd import Variable
import json
from azureml.core.run import Run
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from sklearn.utils.fixes import signature
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

class Predictor():
    def __init__(self, model_folder):
        self.model_path = model_folder
        # config file must be loaded to init a model
        with open(os.path.join(self.model_path, 'config.pkl'), 'rb') as f:
            config = pickle.load(f)
        self.config = config
        self.model = TextCNN(config)
        # model weight file must be loaded to get learnt weight.
        model_file = os.path.join(model_folder, "best_steps_100.pt")
        if model_file is not None:
            self.model.load_state_dict(torch.load(model_file))
        else:
            raise FileNotFoundError("Model File Not Exist")
        # user may set device id here, but now let us ignore it.
        if torch.cuda.is_available() and config.cuda:
            self.model = self.model.cuda()
        self.model.eval()

    def predict(self, data_frame):
        output_label = []
        output_prob = []
        with torch.no_grad():  # ask not to cal
            for index, row in data_frame.iterrows():
                input_setence = row[
                    'text_id']  # I have no idea how to pass the 'text' info to the predict. maybe store in somewhere.
                # input sentence is an int list
                # how to know what kind of tensor to convert?
                x = Variable(torch.LongTensor([input_setence]))
                if torch.cuda.is_available() and self.config.cuda:
                    x = x.cuda()
                output = self.model(x)
                # general classification logic to generate scored label
                probability, predicted = torch.max(output, 1)
                output_label.append(predicted.view(1).cpu().numpy()[0])
                output_prob.append(probability.view(1).cpu().numpy()[0])
            data_frame['Scored Label'] = output_label
            data_frame['Scored Prob'] = output_prob
        return data_frame

    def evaluation(self, df_true, df_predict, df_prob, output_eval_dir):
        run = Run.get_context()

        # precition-recall-curve
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
        run.log_image("precition/recall curve", plot=f1_plt)
        f1_plt.savefig(os.path.join(output_eval_dir, 'precition_recall.png'))

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
        run.log_image("scores", plot=f2_plt)
        f2_plt.savefig(os.path.join(output_eval_dir, 'scores.png'))

        f3_plt = plt.figure(3)
        # Compute fpr, tpr, thresholds and roc auc
        fpr, tpr, thresholds = roc_curve(df_true, df_prob)
        roc_auc = auc(df_true, df_prob)

        plt.plot(fpr, tpr, label='ROC curve (area = %0.3f)' % roc_auc)
        plt.plot([0, 1], [0, 1], 'k--')  # random predictions curve
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate or (1 - Specifity)')
        plt.ylabel('True Positive Rate or (Sensitivity)')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        run.log_image("ROC curve", plot=f3_plt)
        f3_plt.savefig(os.path.join(output_eval_dir, 'roc.png'))

if __name__ == '__main__':
    args = predict_args()
    predictor = Predictor(args.trained_model)
    if not os.path.isdir(args.predict_result_path):
        os.makedirs(args.predict_result_path)
    parquet_path = os.path.join(args.predict_path, 'data.dataset.parquet')
    df = pd.read_parquet(parquet_path, engine='pyarrow')
    out_df = predictor.predict(df)
    out_df.to_parquet(os.path.join(args.predict_result_path, 'data.dataset.parquet'))

    headers = df.columns.values.tolist()
    if 'label' in headers:
        predictor.evaluation(df['label'], out_df['Scored Label'], out_df['Scored Prob'], args.predict_result_path)


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
    with open(os.path.join(args.predict_result_path, 'data_type.json'), 'w') as f:
        json.dump(dct, f)
