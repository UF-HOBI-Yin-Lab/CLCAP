from tkinter import X
import pandas as pd
import random
import torch
import numpy as np
import math


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def reshape_to_linear(x):
    output = np.reshape(x, (x.shape[0], -1))
    return output


def convert_to_numpy(x, y):
    return np.array(x), np.array(y)

# convert distance values into label, distance>=4 (label=1) distance<4 (label=0)


def calculate_label(antigenic_data):
    distance_label = []
    for i in range(0, antigenic_data.shape[0]):
        if antigenic_data['Distance'].iloc[i] >= 4.0:
            distance_label.append(1)
        else:
            distance_label.append(0)
    return distance_label

# extract sequences to construct new dataset


def strain_selection(distance_data, seq_data):
    raw_data = []
    strain1 = []
    strain2 = []
    label = calculate_label(distance_data)
    for i in range(0, distance_data.shape[0]):
        seq1 = []
        seq2 = []
        flag1 = 0
        flag2 = 0
        for j in range(0, seq_data.shape[0]):
            if str(seq_data['description'].iloc[j]).upper() == str(distance_data['Strain1'].iloc[i]).upper():
                seq1 = str(seq_data['seq'].iloc[j]).upper()
                flag1 = 1
            if str(seq_data['description'].iloc[j]).upper() == str(distance_data['Strain2'].iloc[i]).upper():
                seq2 = str(seq_data['seq'].iloc[j]).upper()
                flag2 = 1
            if flag1 == 1 and flag2 == 1:
                break
        strain1.append(seq1)
        strain2.append(seq2)

    raw_data.append(strain1)
    raw_data.append(strain2)
    raw_data.append(label)
    return raw_data


def sample_dataframe(dataframe, ratio):
    df1 = dataframe.sample(frac=ratio)
    df2 = dataframe[~dataframe.index.isin(df1.index)]
    return df1, df2

# split data into training and testing


def train_test_split_data(feature, label, split_ratio):
    setup_seed(20)
    train_x, test_x, train_y, test_y = [], [], [], []
    feature_new, label_new = [], []
    num_of_training = int(math.floor(len(feature) * (1 - split_ratio)))

    shuffled_index = np.arange(len(feature))
    random.shuffle(shuffled_index)
    for i in range(0, len(feature)):
        feature_new.append(feature[shuffled_index[i]])
        label_new.append(label[shuffled_index[i]])

    train_x = feature_new[:num_of_training]
    train_y = label_new[:num_of_training]
    test_x = feature_new[num_of_training:]
    test_y = label_new[num_of_training:]

    train_x = np.array(train_x)
    train_y = np.array(train_y)
    test_x = np.array(test_x)
    test_y = np.array(test_y)

    return train_x, test_x, train_y, test_y


def cnn_training_data(Antigenic_dist, seq):
    raw_data = strain_selection(Antigenic_dist, seq)
    # replace unambiguous with substitutions
    Btworandom = 'DN'
    Jtworandom = 'IL'
    Ztworandom = 'EQ'
    Xallrandom = 'ACDEFGHIKLMNPQRSTVWY'
    for i in range(0, 2):
        for j in range(0, len(raw_data[0])):
            seq = raw_data[i][j]
            seq = seq.replace('B', random.choice(Btworandom))
            seq = seq.replace('J', random.choice(Jtworandom))
            seq = seq.replace('Z', random.choice(Ztworandom))
            seq = seq.replace('X', random.choice(Xallrandom))
            raw_data[i][j] = seq
    # embedding with ProVect
    df = pd.read_csv('protVec_100d_3grams.csv', delimiter='\t')
    trigrams = list(df['words'])
    trigram_to_idx = {trigram: i for i, trigram in enumerate(trigrams)}
    trigram_vecs = df.loc[:, df.columns != 'words'].values

    feature = []
    label = raw_data[2]
    for i in range(0, len(raw_data[0])):
        trigram1 = []
        trigram2 = []
        strain_embedding = []
        seq1 = raw_data[0][i]
        seq2 = raw_data[1][i]
        for j in range(0, len(raw_data[0][0])-2):
            trigram1 = seq1[j:j+3]
            if trigram1[0] == '-' or trigram1[1] == '-' or trigram1[2] == '-':
                tri1_embedding = trigram_vecs[trigram_to_idx['<unk>']]
            else:
                tri1_embedding = trigram_vecs[trigram_to_idx[trigram1]]

            trigram2 = seq2[j:j+3]
            if trigram2[0] == '-' or trigram2[1] == '-' or trigram2[2] == '-':
                tri2_embedding = trigram_vecs[trigram_to_idx['<unk>']]
            else:
                tri2_embedding = trigram_vecs[trigram_to_idx[trigram2]]

            # tri_embedding = tri1_embedding - tri2_embedding
            tri_embedding = tri2_embedding - tri1_embedding
            strain_embedding.append(tri_embedding)

        feature.append(strain_embedding)
    return np.array(feature), np.array(label)
