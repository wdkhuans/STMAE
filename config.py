import json
from typing import NamedTuple
import os

class DatasetConfig(NamedTuple):
    """ Hyperparameters for training """

    node_num: int = 0
    node_dim: int = 0

    sr: int = 0  # sampling rate
    # dataset = Narray with shape (size, seq_len, dimension)
    size: int = 0  # data sample number
    seq_len: int = 0  # seq length
    dimension: int = 0  # feature dimension

    activity_label_index: int = -1  # index of activity label
    activity_label_size: int = 0  # number of activity label
    activity_label: list = []  # names of activity label.

    user_label_index: int = -1  # index of user label
    user_label_size: int = 0  # number of user label

    position_label_index: int = -1  # index of phone position label
    position_label_size: int = 0  # number of position label
    position_label: list = []  # names of position label.

    model_label_index: int = -1  # index of phone model label
    model_label_size: int = 0  # number of model label

    @classmethod
    def from_json(cls, js):
        return cls(**js)

def create_io_config(args, dataset_name, version, pretrain_model=None, target='pretrain'):
    data_path = os.path.join('dataset', dataset_name, dataset_name + '_' + version + '/')
    args.data_path = data_path
    save_path = os.path.join('check', args.dataset, args.path, target + "_" + dataset_name + "_" + version)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    args.save_path = os.path.join(save_path, args.save_model)
    if pretrain_model is not None:
        model_path = os.path.join(save_path, pretrain_model)
        args.pretrain_model = model_path
    else:
        args.pretrain_model = None
    return args

def create_train_config(args, dataset, version):
    path = 'config/train_config.json'
    train_config_all = json.load(open(path, "r"))
    name = dataset + "_" + version
    if name in train_config_all:
        args.balance = train_config_all[name]['balance']
        args.batch_size = train_config_all[name]['batch_size']
        args.label_rate = train_config_all[name]['label_rate']
        args.mask_ratio = train_config_all[name]['mask_ratio']
        args.epoch = train_config_all[name]['epoch']
        args.fine_epoch = train_config_all[name]['fine_epoch']
        args.pre_lr = train_config_all[name]['pre_lr']
        args.fine_lr = train_config_all[name]['fine_lr']
        args.len_mask = train_config_all[name]['len_mask']
        args.embed_dim = train_config_all[name]['embed_dim']
        args.depth = train_config_all[name]['depth']
        args.num_heads = train_config_all[name]['num_heads']
        args.mlp_ratio = train_config_all[name]['mlp_ratio']
        args.decoder_embed_dim = train_config_all[name]['decoder_embed_dim']
        args.decoder_depth = train_config_all[name]['decoder_depth']
        args.decoder_num_heads = train_config_all[name]['decoder_num_heads']
        return args
    else:
        print("No Train Config Found!")
        return args

def load_dataset_stats(dataset, version):
    path = 'config/data_config.json'
    dataset_config_all = json.load(open(path, "r"))
    name = dataset + "_" + version
    if name in dataset_config_all:
        return DatasetConfig.from_json(dataset_config_all[name])
    else:
        return None