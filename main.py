import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from utils import *
from models import fetch_classifier
import train

def pre_train(args, data_train_u, label_train_u, data_valid, label_valid):
    data_set_train = IMUDataset(data_train_u, label_train_u)
    data_set_valid = IMUDataset(data_valid, label_valid)
    data_loader_train = DataLoader(data_set_train, shuffle=True, batch_size=args.batch_size)
    data_loader_valid = DataLoader(data_set_valid, shuffle=False, batch_size=args.batch_size)

    criterion = nn.MSELoss(reduction='none')
    model = fetch_classifier('STMAE_Pre', args=args)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.pre_lr)
    trainer = train.Trainer(model, optimizer, args.save_path, get_device(args.gpu), args)

    def func_loss(model, batch):
        data, _ = batch
        seqs, seq_recon = model(data)
        loss = criterion(seq_recon, seqs)
        return loss

    def func_forward(model, batch):
        data, _ = batch
        seqs, seq_recon = model(data)
        return seq_recon, seqs

    def func_evaluate(seqs, predict_seqs):
        loss_lm = criterion(predict_seqs, seqs)
        return loss_lm.mean().cpu().numpy()

    log_path = os.path.join('check', args.dataset, args.path)
    writer = SummaryWriter(log_path)

    trainer.pretrain(func_loss, func_forward, func_evaluate, data_loader_train, data_loader_valid, 
                     model_file=args.pretrain_model, writer=writer)

def fine_tuning(args, data_train_l, label_train_l, data_valid, label_valid, data_test, label_test):
    data_set_train = IMUDataset(data_train_l, label_train_l)
    data_set_valid = IMUDataset(data_valid, label_valid)
    data_set_test = IMUDataset(data_test, label_test)
    data_loader_train = DataLoader(data_set_train, shuffle=True, batch_size=args.batch_size)
    data_loader_valid = DataLoader(data_set_valid, shuffle=False, batch_size=args.batch_size)
    data_loader_test = DataLoader(data_set_test, shuffle=False, batch_size=args.batch_size)

    criterion = nn.CrossEntropyLoss()
    model = fetch_classifier("STMAE_Finetune", args=args)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.fine_lr)
    trainer = train.Trainer(model, optimizer, args.save_path, get_device(args.gpu), args)

    def func_loss(model, batch):
        inputs, label = batch
        logits = model(inputs)
        loss = criterion(logits, label)
        return loss

    def func_forward(model, batch):
        inputs, label = batch
        logits = model(inputs)
        return logits, label

    def func_evaluate(label, predicts):
        stat = stat_acc_f1(label.cpu().numpy(), predicts.cpu().numpy())
        return stat

    log_path = os.path.join('check', args.dataset, args.path)
    writer = SummaryWriter(log_path)

    trainer.fine_tuning(func_loss, func_forward, func_evaluate, data_loader_train, data_loader_valid, data_loader_test, 
                  model_file=args.pretrain_model, writer=writer)

if __name__ == "__main__":
    args = handle_argv_pre_train()
    data_train_u, label_train_u, data_train_l, label_train_l, data_valid, label_valid, data_test, label_test = load_data(args)

    print("start pre-train\n")
    pre_train(args, data_train_u, label_train_u, data_valid, label_valid)

    print("start fine-tuning\n")
    args = handle_argv_finetune(args)
    fine_tuning(args, data_train_l, label_train_l, data_valid, label_valid, data_test, label_test)

    print("dataset:{}, test_user: {}, path: {}".format(args.dataset, args.test_user, args.path))

    