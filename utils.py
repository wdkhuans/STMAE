import argparse
from torch.utils.data import Dataset
from config import create_io_config, create_train_config, load_dataset_stats
from sklearn.metrics import f1_score

""" Utils Functions """

import random

import numpy as np
import torch
import sys
import time
import os

class IMUDataset(Dataset):
    """ Load sentence pair (sequential or random order) from corpus """
    def __init__(self, data, labels, pipeline=[]):
        super().__init__()
        self.pipeline = pipeline
        self.data = data
        self.labels = labels

    def __getitem__(self, index):
        instance = self.data[index]
        for proc in self.pipeline:
            instance = proc(instance)
        return torch.from_numpy(instance).float(), torch.from_numpy(np.array(self.labels[index])).long()

    def __len__(self):
        return len(self.data)

def set_seeds(seed):
    "set random seeds"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def get_device(gpu):
    "get device (CPU or GPU)"
    if gpu is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cuda:" + gpu if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    print("%s (%d GPUs)" % (device, n_gpu))
    return device

def stat_acc_f1(label, results_estimated):
    label_estimated = np.argmax(results_estimated, 1)
    f1 = f1_score(label, label_estimated, average='macro')
    acc = np.sum(label == label_estimated) / label.size
    return acc, f1

def load_data(args):
    train_data   = np.empty( [0, args.dataset_cfg.seq_len, args.dataset_cfg.dimension], dtype=np.float )
    test_data    = np.empty( [0, args.dataset_cfg.seq_len, args.dataset_cfg.dimension], dtype=np.float )
    train_label  = np.empty( [0], dtype=np.int )
    test_label   = np.empty( [0], dtype=np.int )
    
    for i in range(args.dataset_cfg.user_label_size):
        data = np.load(args.data_path +  'sub_{}_data.npy'.format(i)).astype(np.float32)
        label = np.load(args.data_path +  'sub_{}_label.npy'.format(i)).astype(np.float32)
        if i in args.test_user:
            test_data = np.concatenate( (test_data, data), axis=0 )
            test_label = np.concatenate( (test_label, label), axis=0 )
            print('user for test in finetune: user_{}'.format(i))
        else:
            train_data = np.concatenate( (train_data, data), axis=0 )
            train_label = np.concatenate( (train_label, label), axis=0 )
    
    data_train, label_train, data_valid, label_valid = prepare_simple_dataset(train_data, train_label, training_rate=0.9)
    data_train_labeled, label_train_labeled, data_train_unlabeled, label_train_unlabeled = prepare_simple_dataset(data_train, label_train, training_rate=args.label_rate)

    print('Label Rate: %.4f, Unlabeled Train Size: %d, Labeled Train Size: %d, Valid Size: %d, Test Size: %d.' 
          % (args.label_rate, len(data_train_unlabeled), len(data_train_labeled), len(data_valid), len(test_data)))

    return data_train_unlabeled, label_train_unlabeled, data_train_labeled, label_train_labeled, data_valid, label_valid, test_data, test_label

def prepare_simple_dataset(data, labels, training_rate=0.2):
    arr = np.arange(data.shape[0])
    np.random.shuffle(arr)
    data = data[arr]
    labels = labels[arr]
    train_num = int(data.shape[0] * training_rate)
    data_train = data[:train_num, ...]
    data_test = data[train_num:, ...]
    t = np.min(labels)
    label_train = labels[:train_num] - t
    label_test = labels[train_num:] - t

    return data_train, label_train, data_test, label_test

def handle_argv_pre_train():
    parser = argparse.ArgumentParser(description='STMAE: Spatial-Temporal Masked Autoencoder for Multi-Device Wearable Human Activity Recognition')
    parser.add_argument('-u', '--test_user', type=int, nargs='+', default=[0], help='Test user')
    parser.add_argument('-d', '--dataset', type=str, default='opp', help='Dataset name', choices=['realworld', 'opp', 'realdisp', 'pamap'])
    parser.add_argument('-g', '--gpu', type=str, default=None, help='Set specific GPU')
    parser.add_argument('-f', '--model_file', type=str, default=None, help='Pretrain model file')
    parser.add_argument('-l', '--label_index', type=int, default=23, help='Label Index')
    parser.add_argument('-s', '--seed', type=int, default=1, help='Seed')
    parser.add_argument('-m', '--save_model', type=str, default='model', help='The saved model name')
    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)

    if args.dataset == 'realworld':
        args.dataset_version = '50_150'
    elif args.dataset == 'opp':
        args.dataset_version = '30_60'
    elif args.dataset == 'pamap':
         args.dataset_version = '100_300'
    elif args.dataset == 'realdisp':
        args.dataset_version = '50_150'

    time_now = time.localtime(time.time())
    args.path = str(time_now.tm_mon)+'m_'+str(time_now.tm_mday)+'d_'+str(time_now.tm_hour)+'h_'+str(time_now.tm_min)+'m_'+str(time_now.tm_sec)+'s' 
    args.path = args.path + '_'+ str(random.randint(0, 100))
    os.mkdir(os.path.join('check', args.dataset, args.path))

    args.dataset_cfg = load_dataset_stats(args.dataset, args.dataset_version)
    args = create_io_config(args, args.dataset, args.dataset_version, pretrain_model=args.model_file)
    args = create_train_config(args, args.dataset, args.dataset_version)
    set_seeds(args.seed)
    print(args)
    return args

def handle_argv_finetune(args):
    save_path = os.path.join('check', args.dataset, args.path, 'finetune_' + args.dataset + "_" + args.dataset_version)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    args.save_path = os.path.join(save_path, args.save_model)
    args.pretrain_model = os.path.join('check', args.dataset, args.path, 'pretrain_' + args.dataset + "_" + args.dataset_version, args.save_model)
    return args