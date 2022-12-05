from __future__ import division
from pytorch_metric_learning import losses
from Contrastive import *
from validation import get_time_string
from validation import evaluate
from validation import get_confusion_matrix

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import math
import time

sys.path.append(os.path.abspath("/home/biaoye/IAV-CNN/Updated_data"))


class early_stopper():
    def __init__(self, times, metric_methods, name):
        super(early_stopper, self).__init__()
        self.times = times
        self.metric_methods = metric_methods
        if self.metric_methods == 'acc':
            self.best_score = 0
        else:
            self.best_score = 1000
        self.counting_times = 0
        self.name = name

    def counting(self, model, val_score):
        if self.metric_methods == 'acc':
            if val_score > self.best_score:
                self.counting_times = 0
                self.best_score = val_score
                torch.save(model, self.name)
            else:
                self.counting_times += 1
            if self.counting_times > self.times:
                return True
            return False
        else:
            if val_score < self.best_score:
                self.counting_times = 0
                self.best_score = val_score
                torch.save(model, self.name)
            else:
                self.counting_times += 1
            if self.counting_times > self.times:
                return True
            return False

    def load_best_model(self, model, val_score):
        if self.counting(model, val_score):
            model = torch.load(self.name)
            return model, True
        return model, False


def predictions_from_output(scores):
    """
    Maps logits to class predictions.
    """
    prob = F.softmax(scores, dim=1)
    _, predictions = prob.topk(1)
    return predictions


def verify_model(model, X, Y, batch_size):
    """
    Checks the loss at initialization of the model and asserts that the
    training examples in a batch aren't mixed together by backpropagating.
    """
    print('Sanity checks:')
    criterion = torch.nn.CrossEntropyLoss()
    scores, _ = model(X, model.init_hidden(Y.shape[0]))
    print(' Loss @ init %.3f, expected ~%.3f' %
          (criterion(scores, Y).item(), -math.log(1 / model.output_dim)))

    mini_batch_X = X[:, :batch_size, :]
    mini_batch_X.requires_grad_()
    criterion = torch.nn.MSELoss()
    scores, _ = model(mini_batch_X, model.init_hidden(batch_size))

    non_zero_idx = 1
    perfect_scores = [[0, 0] for i in range(batch_size)]
    not_perfect_scores = [[1, 1] if i == non_zero_idx else [0, 0]
                          for i in range(batch_size)]

    scores.data = torch.FloatTensor(not_perfect_scores)
    Y_perfect = torch.FloatTensor(perfect_scores)
    loss = criterion(scores, Y_perfect)
    loss.backward()

    zero_tensor = torch.FloatTensor([0] * X.shape[2])
    for i in range(mini_batch_X.shape[0]):
        for j in range(mini_batch_X.shape[1]):
            if sum(mini_batch_X.grad[i, j] != zero_tensor):
                assert j == non_zero_idx, 'Input with loss set to zero has non-zero gradient.'

    mini_batch_X.detach()
    print('Backpropagated dependencies OK')


def Max_Margin_Training(model, optimizer, criterion, TrainX, TrainY, batch_size, epochs):
    start_time = time.time()
    num_of_examples = TrainX.shape[0]
    stopper = early_stopper(times=5, metric_methods='loss',
                            name='/home/biaoye/IAV-CNN/code/saved_model/best_contrastive_model.pt')
    for epoch in range(epochs):

        model.train()
        train_loss = 0
        for count in range(0, num_of_examples - batch_size + 1, batch_size):

            X_batch = TrainX[count:count+batch_size, :, :, :]
            Y_batch = TrainY[count:count+batch_size]

            X_batch = torch.tensor(X_batch, dtype=torch.float32).cuda()
            Y_batch = torch.tensor(Y_batch, dtype=torch.int64).cuda()

            scores = model.forward_contrastive(X_batch)
            loss = criterion(scores, Y_batch)

            train_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        elapsed_time = time.time() - start_time
        train_loss = np.mean(train_loss)/int(num_of_examples/batch_size)
        model, flag = stopper.load_best_model(model, train_loss)
        print('Contrastive Training Epoch %d Time %s' %
              (epoch, get_time_string(elapsed_time)))
        print(f'Contrastive Training Loss {train_loss}')
        if flag:
            break


def NT_Xnet_Training(model, optimizer, criterion, TrainX, TrainY, batch_size, epochs):
    start_time = time.time()
    num_of_examples = TrainX.shape[0]
    stopper = early_stopper(times=5, metric_methods='loss',
                            name='/home/biaoye/IAV-CNN/code/saved_model/best_contrastive_model.pt')
    for epoch in range(epochs):

        model.train()
        train_loss = 0
        for count in range(0, num_of_examples - batch_size + 1, batch_size):
            X_batch = TrainX[count:count+batch_size, :, :, :]
            Y_batch = TrainY[count:count+batch_size]

            X_batch = NTXent_data(X_batch)

            X_batch = torch.cat(X_batch)
            Y_batch = Y_batch.repeat(2)

            X_batch = torch.tensor(X_batch, dtype=torch.float32).cuda()
            Y_batch = torch.tensor(Y_batch, dtype=torch.int64).cuda()

            scores = model.forward_contrastive(X_batch)
            loss = criterion(scores, Y_batch)

            train_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        elapsed_time = time.time() - start_time
        train_loss = np.mean(train_loss)/int(num_of_examples/batch_size)
        model, flag = stopper.load_best_model(model, train_loss)
        print('Contrastive Training Epoch %d Time %s' %
              (epoch, get_time_string(elapsed_time)))
        print(f'Contrastive Training Loss {train_loss}')
        if flag:
            break


def Contrastive_Learning(model, optimizer, loss, X, Y, batch_size, epochs):
    print("Start Contrastive Learning!")
    optimizer = optimizer
    if loss == 'Max_Margin_Loss':
        print('Max_Margin Mode')
        criterion = losses.ContrastiveLoss()
        Max_Margin_Training(model, optimizer, criterion,
                            X, Y, batch_size, epochs)
    if loss == 'NT_Xnet_Loss':
        criterion = SupervisedContrastiveLoss()
        print('SupervisedContrastiveLoss Mode')
        NT_Xnet_Training(model, optimizer, criterion, X, Y, batch_size, epochs)
    print('Contrastive Learning Completed!')


def train_cnn(model, optimizer, criterion, epochs, batch_size, X, Y, X_test, Y_test,
              ctr_start, contrastive, ctr_loss, ctr_batch, ctr_epochs, trainset, seq, Interval):
    """
    Training loop for a model utilizing hidden states.

    verify enables sanity checks of the model.
    epochs decides the number of training iterations.
    learning rate decides how much the weights are updated each iteration.
    batch_size decides how many examples are in each mini batch.
    show_attention decides if attention weights are plotted.
    """
    print_interval = Interval
    optimizer = optimizer
    criterion = criterion
    num_of_examples = X.shape[0]
    num_of_batches = math.floor(num_of_examples/batch_size)
    stopper = early_stopper(times=5, metric_methods='acc',
                            name='/home/biaoye/IAV-CNN/code/saved_model/best_ce_model.pt')
    # if verify:
    # verify_model(model, X, Y, batch_size)

    all_losses = []
    all_val_losses = []
    all_accs = []
    all_pres = []
    all_recs = []
    all_fscores = []
    all_mccs = []
    all_val_accs = []

    best_val = 0

    start_time = time.time()

    for epoch in range(epochs):

        if (epoch == ctr_start) and contrastive:
            contrastive_optimizer = torch.optim.Adam(
                model.parameters(), lr=0.005, weight_decay=1e-5)
            Contrastive_Learning(model, contrastive_optimizer,
                                 ctr_loss, X, Y, ctr_batch, ctr_epochs)
            model.freeze_projection()
            stopper = early_stopper(times=5, metric_methods='acc',
                                    name='/home/biaoye/IAV-CNN/code/saved_model/best_ce_model.pt')
        model.train()
        running_loss = 0
        running_acc = 0
        running_pre = 0
        running_pre_total = 0
        running_rec = 0
        running_rec_total = 0
        epoch_fscore = 0
        running_mcc_numerator = 0
        running_mcc_denominator = 0
        running_rec_total = 0

        for count in range(0, num_of_examples - batch_size + 1, batch_size):
            X_batch = X[count:count+batch_size, :, :, :]
            Y_batch = Y[count:count+batch_size]

            scores = model(X_batch)
            loss = criterion(scores, Y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            predictions = predictions_from_output(scores)
            conf_matrix = get_confusion_matrix(Y_batch, predictions)
            TP, FP, FN, TN = conf_matrix[0][0], conf_matrix[0][1], conf_matrix[1][0], conf_matrix[1][1]
            running_acc += TP + TN
            running_pre += TP
            running_pre_total += TP + FP
            running_rec += TP
            running_rec_total += TP + FN
            running_mcc_numerator += (TP * TN - FP * FN)
            if ((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)) == 0:
                running_mcc_denominator += 0
            else:
                running_mcc_denominator += math.sqrt(
                    (TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
            running_loss += loss.item()

        elapsed_time = time.time() - start_time
        epoch_acc = running_acc / Y.shape[0]
        all_accs.append(epoch_acc)

        if running_pre_total == 0:
            epoch_pre = 0
        else:
            epoch_pre = running_pre / running_pre_total
        all_pres.append(epoch_pre)

        if running_rec_total == 0:
            epoch_rec = 0
        else:
            epoch_rec = running_rec / running_rec_total
        all_recs.append(epoch_rec)

        if (epoch_pre + epoch_rec) == 0:
            epoch_fscore = 0
        else:
            epoch_fscore = 2 * epoch_pre * epoch_rec / (epoch_pre + epoch_rec)
        all_fscores.append(epoch_fscore)

        if running_mcc_denominator == 0:
            epoch_mcc = 0
        else:
            epoch_mcc = running_mcc_numerator / running_mcc_denominator
        all_mccs.append(epoch_mcc)

        epoch_loss = running_loss / num_of_batches
        all_losses.append(epoch_loss)

        with torch.no_grad():
            model.eval()
            test_scores = model(X_test)
            predictions = predictions_from_output(test_scores)
            predictions = predictions.view_as(Y_test)

            precision, recall, fscore, mcc, val_acc = evaluate(
                Y_test, predictions)

            val_loss = criterion(test_scores, Y_test).item()
            all_val_losses.append(val_loss)
            all_val_accs.append(val_acc)

        if (epoch+1) % print_interval == 0:
            print('Epoch %d Time %s' % (epoch, get_time_string(elapsed_time)))
            print('T_loss %.4f\tT_acc %.4f\tT_pre %.4f\tT_rec %.4f\tT_fscore %.4f\tT_mcc %.4f' % (
                epoch_loss, epoch_acc, epoch_pre, epoch_rec, epoch_fscore, epoch_mcc))
            print('V_loss %.4f\tV_acc %.4f\tV_pre %.4f\tV_rec %.4f\tV_fscore %.4f\tV_mcc %.4f' % (
                val_loss, val_acc, precision, recall, fscore, mcc))
        model, flag = stopper.load_best_model(model, val_acc)
        if(val_acc > best_val):
            best_val = val_acc
        if flag:
            break
    return best_val
