import pyarrow.parquet as pq
import os
import pickle

import pandas as pd
import torch
from .TextCNN import TextCNN
from .args_util import predict_args
from torch.autograd import Variable
import json

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
                _, predicted = torch.max(output, 1)
                output_label.append(predicted.view(1).cpu().numpy()[0])
            data_frame['Scored Label'] = output_label
        return data_frame


if __name__ == '__main__':
    args = predict_args()
    predictor = Predictor(args.trained_model)
    if not os.path.isdir(args.predict_result_path):
        os.makedirs(args.predict_result_path)
    parquet_path = os.path.join(args.predict_path, 'data.dataset.parquet')
    df = pd.read_parquet(parquet_path, engine='pyarrow')

    out_df = predictor.predict(df)
    out_df.to_parquet(os.path.join(args.predict_result_path, 'data.dataset.parquet'))


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
