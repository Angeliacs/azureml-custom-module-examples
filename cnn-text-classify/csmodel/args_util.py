import argparse
import datetime
import torch
import os

def train_args():
    """
    获取参数
    :return: 模型参数
    """
    parser = argparse.ArgumentParser(description='CNN text classificer')
    # learning
    parser.add_argument_group("Custom module path args")
    parser.add_argument('--train-file', type=str, default='../data/test_data_folder',
                        help='the train file path')
    parser.add_argument('--test-file', type=str, default='../data/test_data_folder',
                        help='the test file path')
    parser.add_argument('--trained-model', type=str,
                        help='the trained path')
    parser.add_argument('--vocab-path', type=str, default='./vocab/',
                        help='the output path of vocab')
    parser.add_argument('--log-dir', type=str, default='logs',
                        help='where to save the log [tensorboard]')

    parser.add_argument_group("Custom module released parameter args")
    parser.add_argument('--embed-dim', type=int, default=300,
                        help='number of embedding dimension [default: 300]')
    parser.add_argument('--kernel-num', type=int, default=128,
                        help='number of each kind of kernel')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='the probability for dropout [default: 0.5]')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='batch size for training [default: 32]')
    parser.add_argument('--l2', type=float, default=0.,
                        help='initial l2 regularization [default: 0.]')
    parser.add_argument('--test-interval', type=int, default=100,
                        help='how many steps to wait before testing [default: 100]')
    parser.add_argument('--label-column', type=str, default='label',
                        help='Select the column that contains the label or outcome column [default: 0]')
    parser.add_argument('--text-column', type=str, default='text',
                        help='Select the column that contains the Text or input column [default: 2]')

    parser.add_argument_group("Custom module not released parameter args")
    parser.add_argument('--snapshot', type=str,
                        help='where to save the snapshot')
    parser.add_argument('--lr', type=float, default=0.0005,
                        help='initial learning rate [default: 0.001]')

    parser.add_argument('--epochs', type=int, default= 1,
                        help='number of epochs for train [default: 1]')
    parser.add_argument('--max-len', type=int, default=4096,
                        help='max len of sentence for training [default: 1024]')
    parser.add_argument('--log-interval', type=int, default=10,
                        help='how many steps to wait before logging training status [default: 10]')

    parser.add_argument('--save-interval', type=int, default=500,
                        help='how many steps to wait before saving [default:500]')

    parser.add_argument('--early-stop', type=int, default=300,
                        help='iteration numbers to stop without performance increasing')
    parser.add_argument('--save-best', type=bool, default=True,
                        help='whether to save when get best performance')
    # data
    parser.add_argument('--test-ratio', type=float, default=0.3,
                        help='test data ratio')
    parser.add_argument('--shuffle', action='store_true', default=True,
                        help='shuffle the data every epoch')
    # model


    parser.add_argument('--hidden-dim', type=int, default=128,
                        help='hidden dimension [default: 128]')

    parser.add_argument('--kernel-sizes', type=str, default='2,4,7',
                        help='comma-separated kernel size to use for convolution')

    parser.add_argument('--static', action='store_true', default=False,
                        help='fix the embedding')
    # device
    parser.add_argument('--device', type=int, default=0,
                        help='device to use for iterate data, -1 mean cpu [default: -1]')
    parser.add_argument('--cuda', action='store_true', default=True,
                        help='disable the gpu')


    args, _ = parser.parse_known_args()
    args.cuda = args.cuda and torch.cuda.is_available()
    now_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    # args.save_dir = os.path.join(args.save_dir, now_time)
    args.log_dir = os.path.join(args.log_dir, now_time)
    return args

def predict_args():
    parser = argparse.ArgumentParser(description='Predict Args')

    parser.add_argument('--trained-model', type=str,
                        help='the trained path')
    parser.add_argument('--predict-path', type=str, default="debug_out/word_id/",
                        help='the test dataset path')
    parser.add_argument('--predict-result-path', type=str, default="debug_out/predict_res/" ,
                        help='the predicted output path')
    args, _ = parser.parse_known_args()
    return args

def preprocess_args():
    parser = argparse.ArgumentParser(description='Preprocess Args')

    parser.add_argument('--input-data', type=str, default='../data/test_data_folder',
                        help='the input data path')
    parser.add_argument('--input-vocab', type=str, default='debug_out/vocab/',
                        help='the vocab path')
    parser.add_argument('--output-data', type=str, default='debug_out/word_id/',
                        help='the output data path')
    args, _ = parser.parse_known_args()
    return args

def print_parameters(args):
    """
    print model parameters
    :param args: model parameters
    :return: None
    """
    print("\nParameters:")
    for attr, value in sorted(args.__dict__.items()):
        print("\t{}={}".format(attr.upper(), value))

