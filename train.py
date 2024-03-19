import copy
from tkinter import NO
import torch
import torch.nn as nn

from timm.models.layers import trunc_normal_

class Trainer(object):
    """Training Helper Class"""
    def __init__(self, model, optimizer, save_path, device, args=None):
        self.model = model
        self.optimizer = optimizer
        self.save_path = save_path
        self.device = device
        self.args = args

    def pretrain(self, func_loss, func_forward, func_evaluate, 
                 data_loader_train, data_loader_vali, 
                 model_file=None, data_parallel=False, writer=None):
        """ Train Loop """
        self.load(model_file)
        model = self.model.to(self.device)
        if data_parallel: # use Data Parallelism with Multi-GPU
            model = nn.DataParallel(model)

        global_step = 0 # global iteration steps regardless of epochs
        best_loss = 1e6
        model_best = model.state_dict()

        for e in range(self.args.epoch):
            loss_sum = 0.0 # the sum of iteration losses to get average loss in every epoch
            self.model.train()
            for _, batch in enumerate(data_loader_train):
                batch = [t.to(self.device) for t in batch]
                self.optimizer.zero_grad()
                loss = func_loss(model, batch)  

                loss = loss.mean()
                loss.backward()
                self.optimizer.step()
                global_step += 1
                loss_sum += loss.item()

            loss_eva = self.run(func_forward, func_evaluate, data_loader_vali)
            print('Epoch %d/%d : Train Loss %5.4f. Valid Loss %5.4f'
                    % (e + 1, self.args.epoch, loss_sum / len(data_loader_train), loss_eva))
            writer.add_scalar('pre_loss/loss_train', loss_sum / len(data_loader_train), global_step=e + 1)
            writer.add_scalar('pre_loss/loss_eva', loss_eva, global_step=e + 1)
            if loss_eva < best_loss:
                best_loss = loss_eva
                model_best = copy.deepcopy(model.state_dict())
                self.save(0)

        self.model.load_state_dict(model_best)
        print('The Total Epoch have been reached.')

    def fine_tuning(self, func_loss, func_forward, func_evaluate, 
                    data_loader_train, data_loader_valid, data_loader_test, 
                    model_file=None, data_parallel=False, writer=None):
        """ Train Loop """
        self.load(model_file)
        model = self.model.to(self.device)
        if data_parallel: # use Data Parallelism with Multi-GPU
            model = nn.DataParallel(model)

        global_step = 0 # global iteration steps regardless of epochs
        vali_f1_best = 0.0
        model_best = model.state_dict()

        for e in range(self.args.fine_epoch):
            loss_sum = 0.0 # the sum of iteration losses to get average loss in every epoch
            self.model.train()
            for _, batch in enumerate(data_loader_train):
                batch = [t.to(self.device) for t in batch]
                self.optimizer.zero_grad()
                loss = func_loss(model, batch)

                loss = loss.mean()
                loss.backward()
                self.optimizer.step()
                global_step += 1
                loss_sum += loss.item()

            train_acc, train_f1 = self.run(func_forward, func_evaluate, data_loader_train)
            vali_acc, vali_f1, loss_vali = self.run(func_forward, func_evaluate, data_loader_valid, func_loss=func_loss)
            test_acc, test_f1, loss_test = self.run(func_forward, func_evaluate, data_loader_test, func_loss=func_loss)
            print('Epoch %d/%d : Average Loss %5.4f/%5.4f/%5.4f, Accuracy: %6.4f/%6.4f/%6.4f, F1: %6.4f/%6.4f/%6.4f'
                  % (e+1, self.args.fine_epoch, loss_sum / len(data_loader_train), loss_vali, loss_test, 
                     train_acc*100, vali_acc*100, test_acc*100, train_f1*100, vali_f1*100, test_f1*100))
            
            writer.add_scalar('loss/loss_train', loss_sum / len(data_loader_train), global_step=e + 1)
            writer.add_scalar('loss/loss_vali', loss_vali, global_step=e + 1)
            writer.add_scalar('loss/loss_test', loss_test, global_step=e + 1)
            writer.add_scalar('acc/train_acc', train_acc*100, global_step=e + 1)
            writer.add_scalar('acc/vali_acc', vali_acc*100, global_step=e + 1)
            writer.add_scalar('acc/test_acc', test_acc*100, global_step=e + 1)
            writer.add_scalar('f1/train_f1', train_f1*100, global_step=e + 1)
            writer.add_scalar('f1/vali_f1', vali_f1*100, global_step=e + 1)
            writer.add_scalar('f1/test_f1', test_f1*100, global_step=e + 1)

            if vali_f1 > vali_f1_best:
                vali_f1_best = vali_f1
                model_best = copy.deepcopy(model.state_dict())
                self.save(0)

        self.model.load_state_dict(model_best)
        print('The Total Epoch have been reached.')

    def run(self, func_forward, func_evaluate, data_loader, model_file=None, data_parallel=False, func_loss=None):
        """ Evaluation Loop """
        self.model.eval() # evaluation mode
        self.load(model_file)
        model = self.model.to(self.device)
        if data_parallel: # use Data Parallelism with Multi-GPU
            model = nn.DataParallel(model)

        if func_loss:
            results = [] # prediction results
            labels = []
            loss_sum = 0.0
            for batch in data_loader:
                batch = [t.to(self.device) for t in batch]
                with torch.no_grad(): # evaluation without gradient calculation
                    result, label = func_forward(model, batch)
                    results.append(result)
                    labels.append(label)
                    loss = func_loss(model, batch)
                    loss = loss.mean()
                    loss_sum += loss.item()
            loss_data = loss_sum / len(data_loader)

            data_acc, data_f1 = func_evaluate(torch.cat(labels, 0), torch.cat(results, 0))
            return data_acc, data_f1, loss_data

        else:
            results = [] # prediction results
            labels = []
            for batch in data_loader:
                batch = [t.to(self.device) for t in batch]
                with torch.no_grad(): # evaluation without gradient calculation
                    result, label = func_forward(model, batch)
                    results.append(result)
                    labels.append(label)
            if func_evaluate:
                return func_evaluate(torch.cat(labels, 0), torch.cat(results, 0))
            else:
                return torch.cat(results, 0).cpu().numpy()

    def load(self, model_file=None):
        """ load saved model or pretrained transformer (a part of model) """
        if model_file:
            checkpoint = torch.load(model_file + '.pt', map_location=self.device)
            print("Load pre-trained checkpoint from: %s" % model_file)
            checkpoint_model = checkpoint
            state_dict = self.model.state_dict()
            for k in ['head.weight', 'head.bias']:
                if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                    print(f"Removing key {k} from pretrained checkpoint")
                    del checkpoint_model[k]
            # load pre-trained model
            msg = self.model.load_state_dict(checkpoint_model, strict=False)
            print(msg)
            trunc_normal_(self.model.head.weight, std=2e-5)
            for _, p in self.model.named_parameters():
                p.requires_grad = False
            for _, p in self.model.head.named_parameters():
                p.requires_grad = True
            return

    def save(self, i=0):
        """ save current model """
        if i != 0:
            torch.save(self.model.state_dict(), self.save_path + "_" + str(i) + '.pt')
        else:
            torch.save(self.model.state_dict(),  self.save_path + '.pt')

