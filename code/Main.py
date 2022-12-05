# -*- coding: utf-8 -*-
import os
import sys
from random import sample
import numpy as np
import pandas as pd
import torch
import warnings
from data_generation import cnn_training_data, sample_dataframe, reshape_to_linear, convert_to_numpy
from model import *
from Contrastive import *
from train_cnn import train_cnn
from data_generation import setup_seed

warnings.filterwarnings('ignore')
os.chdir("/home/biaoye/CL_CAP/Updated_data")
sys.path.append(os.path.abspath("/home/biaoye/CL_CAP/Updated_data"))


def main():
    parameters = {

        # Random Seed,
        'seed': 100,

        # select influenza subtype，ex: ‘H1N1’, 'H3N2','H5N1'
        'subtype': 'H1N1',

        # 'rf_baseline', 'lr_baseline', 'knn_baseline', 'svm_baseline', 'nn_baseline', 'cnn','iav-cnn'
        'model': ['cnn'],

        # Number of hidden units in the encoder
        'hidden_size': 64,

        # Droprate (applied at input)
        'dropout': 0.9,

        # Note, no learning rate decay implemented
        'learning_rate': 0.001,

        # Size of mini batch
        'batch_size': 32,

        # Number of training iterations
        'num_of_epochs': 100,

        # Displaying Interval
        'Interval': 1,

        # Contrastive Learning
        'Contrastive': False,

        # Contrastive Learning Dimension,
        'ctr_dim': 64,

        # Contrastive Learning Loss, Max_Margin_Loss, NT_Xnet_Loss
        'ctr_loss': 'NT_Xnet_Loss',

        # Contrastive Start epoch,
        'ctr_start': 0,

        # Contrastive Learning epoch,
        'ctr_epoch': 100,

        # Contrastive Batch Size,
        'ctr_batch': 16
    }
    # read antigenic data and sequence data
    baselines = parameters['model']
    Antigenic_dist = pd.read_csv(
        f"antigenic/{parameters['subtype']}_antigenic.csv")
    seq = pd.read_csv(
        f"sequence/{parameters['subtype']}/{parameters['subtype']}_sequence_HA1.csv", names=['seq', 'description'])

    setup_seed(parameters['seed'])
    # if using train and test split, should be (1 - 0.8 = 0.2)
    trainset, testset = sample_dataframe(Antigenic_dist, 0.8)
    train_x, train_y = cnn_training_data(trainset, seq)
    test_x, test_y = cnn_training_data(testset, seq)

    for baseline in baselines:
        setup_seed(parameters['seed'])
        if baseline == 'rf_baseline':
            print(f"rf_baseline + ProVect on {parameters['subtype']}:")
            rf_baseline(reshape_to_linear(train_x), train_y,
                        reshape_to_linear(test_x), test_y)
        elif baseline == 'lr_baseline':
            print(f"lr_baseline + ProVect on {parameters['subtype']}:")
            lr_baseline(reshape_to_linear(train_x), train_y,
                        reshape_to_linear(test_x), test_y)
        elif baseline == 'svm_baseline':
            print(f"svm_baseline + ProVect on {parameters['subtype']}:")
            svm_baseline(reshape_to_linear(train_x), train_y,
                         reshape_to_linear(test_x), test_y)
        elif baseline == 'knn_baseline':
            print(f"knn_baseline + ProVect on {parameters['subtype']}:")
            knn_baseline(reshape_to_linear(train_x), train_y,
                         reshape_to_linear(test_x), test_y)
        elif baseline == 'nn_baseline':
            print(f"nn_baseline + ProVect on {parameters['subtype']}:")
            nn_baseline(reshape_to_linear(train_x), train_y,
                        reshape_to_linear(test_x), test_y)
        elif baseline == 'cnn' or baseline == 'iav-cnn':
            print(f"{baseline} + ProVect on {parameters['subtype']}:")
            train_x = np.reshape(train_x, (np.array(train_x).shape[0], 1, np.array(
                train_x).shape[1], np.array(train_x).shape[2]))
            test_x = np.reshape(test_x, (np.array(test_x).shape[0], 1, np.array(
                test_x).shape[1], np.array(test_x).shape[2]))
            print(np.array(train_x).shape)
            print(np.array(test_x).shape)

            train_x = torch.tensor(train_x, dtype=torch.float32)
            train_y = torch.tensor(train_y, dtype=torch.int64)
            test_x = torch.tensor(test_x, dtype=torch.float32)
            test_y = torch.tensor(test_y, dtype=torch.int64)
            if baseline == 'cnn':
                net = CNN(parameters['subtype'],
                          parameters['dropout'], parameters['ctr_dim'])
            else:
                net = IAV_CNN(
                    1, 128, 1, 2, parameters['ctr_dim'], parameters['dropout'])
            if torch.cuda.is_available():
                print('running with GPU')
                net.cuda()
                train_x = torch.tensor(train_x, dtype=torch.float32).cuda()
                train_y = torch.tensor(train_y, dtype=torch.int64).cuda()
                test_x = torch.tensor(test_x, dtype=torch.float32).cuda()
                test_y = torch.tensor(test_y, dtype=torch.int64).cuda()

            optimizer = torch.optim.Adam(
                net.parameters(), lr=parameters['learning_rate'], weight_decay=1e-5)
            # optimizer = torch.optim.SGD(net.parameters(), lr=parameters['learning_rate'],momentum=0.9,weight_decay = 1e-4)
            criterion = torch.nn.CrossEntropyLoss()

            best_val = train_cnn(net, optimizer, criterion, parameters['num_of_epochs'],
                                 parameters['batch_size'], train_x, train_y, test_x, test_y,
                                 parameters['ctr_start'], parameters['Contrastive'], parameters['ctr_loss'],
                                 parameters['ctr_batch'], parameters['ctr_epoch'],
                                 trainset, seq, parameters['Interval'])
            print(f"Final Best Val Acc = {best_val}")


if __name__ == '__main__':
    main()
