import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn import neighbors
from sklearn import ensemble
from sklearn import linear_model
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import matthews_corrcoef
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from validation import evaluate


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def lr_baseline(X, Y, X_test, Y_test, method=None):
    clf = linear_model.LogisticRegression().fit(X, Y)
    train_acc = accuracy_score(Y, clf.predict(X))
    train_pre = precision_score(Y, clf.predict(X))
    train_rec = recall_score(Y, clf.predict(X))
    train_fscore = f1_score(Y, clf.predict(X))
    train_mcc = matthews_corrcoef(Y, clf.predict(X))

    Y_pred = clf.predict(X_test)
    precision, recall, fscore, mcc, val_acc = evaluate(Y_test, Y_pred)
    print('T_acc %.4f\tT_pre %.4f\tT_rec %.4f\tT_fscore %.4f\tT_mcc %.4f'
          % (train_acc, train_pre, train_rec, train_fscore, train_mcc))
    print('V_acc  %.4f\tV_pre %.4f\tV_rec %.4f\tV_fscore %.4f\tV_mcc %.4f'
          % (val_acc, precision, recall, fscore, mcc))


def knn_baseline(X, Y, X_test, Y_test, method=None):
    clf = neighbors.KNeighborsClassifier().fit(X, Y)
    train_acc = accuracy_score(Y, clf.predict(X))
    train_pre = precision_score(Y, clf.predict(X))
    train_rec = recall_score(Y, clf.predict(X))
    train_fscore = f1_score(Y, clf.predict(X))
    train_mcc = matthews_corrcoef(Y, clf.predict(X))

    Y_pred = clf.predict(X_test)
    precision, recall, fscore, mcc, val_acc = evaluate(Y_test, Y_pred)
    print('T_acc %.4f\tT_pre %.4f\tT_rec %.4f\tT_fscore %.4f\tT_mcc %.4f'
          % (train_acc, train_pre, train_rec, train_fscore, train_mcc))
    print('V_acc  %.4f\tV_pre %.4f\tV_rec %.4f\tV_fscore %.4f\tV_mcc %.4f'
          % (val_acc, precision, recall, fscore, mcc))


def svm_baseline(X, Y, X_test, Y_test, method=None):
    clf = SVC(gamma='auto', class_weight='balanced').fit(X, Y)
    train_acc = accuracy_score(Y, clf.predict(X))
    train_pre = precision_score(Y, clf.predict(X))
    train_rec = recall_score(Y, clf.predict(X))
    train_fscore = f1_score(Y, clf.predict(X))
    train_mcc = matthews_corrcoef(Y, clf.predict(X))

    Y_pred = clf.predict(X_test)
    precision, recall, fscore, mcc, val_acc = evaluate(Y_test, Y_pred)
    print('T_acc %.4f\tT_pre %.4f\tT_rec %.4f\tT_fscore %.4f\tT_mcc %.4f'
          % (train_acc, train_pre, train_rec, train_fscore, train_mcc))
    print('V_acc  %.4f\tV_pre %.4f\tV_rec %.4f\tV_fscore %.4f\tV_mcc %.4f'
          % (val_acc, precision, recall, fscore, mcc))


def rf_baseline(X, Y, X_test, Y_test):
    clf = ensemble.RandomForestClassifier().fit(X, Y)
    train_acc = accuracy_score(Y, clf.predict(X))
    train_pre = precision_score(Y, clf.predict(X))
    train_rec = recall_score(Y, clf.predict(X))
    train_fscore = f1_score(Y, clf.predict(X))
    train_mcc = matthews_corrcoef(Y, clf.predict(X))

    Y_pred = clf.predict(X_test)
    precision, recall, fscore, mcc, val_acc = evaluate(Y_test, Y_pred)
    print('T_acc %.4f\tT_pre %.4f\tT_rec %.4f\tT_fscore %.4f\tT_mcc %.4f'
          % (train_acc, train_pre, train_rec, train_fscore, train_mcc))
    print('V_acc  %.4f\tV_pre %.4f\tV_rec %.4f\tV_fscore %.4f\tV_mcc %.4f'
          % (val_acc, precision, recall, fscore, mcc))


def nn_baseline(X, Y, X_test, Y_test):
    clf = MLPClassifier(random_state=100).fit(X, Y)
    train_acc = accuracy_score(Y, clf.predict(X))
    train_pre = precision_score(Y, clf.predict(X))
    train_rec = recall_score(Y, clf.predict(X))
    train_fscore = f1_score(Y, clf.predict(X))
    train_mcc = matthews_corrcoef(Y, clf.predict(X))

    Y_pred = clf.predict(X_test)
    precision, recall, fscore, mcc, val_acc = evaluate(Y_test, Y_pred)
    print('T_acc %.4f\tT_pre %.4f\tT_rec %.4f\tT_fscore %.4f\tT_mcc %.4f'
          % (train_acc, train_pre, train_rec, train_fscore, train_mcc))
    print('V_acc  %.4f\tV_pre %.4f\tV_rec %.4f\tV_fscore %.4f\tV_mcc %.4f'
          % (val_acc, precision, recall, fscore, mcc))


class CNN(nn.Module):
    def __init__(self, virus_type, dropout=0.3, contrastive_dimension=64):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.mp = nn.MaxPool2d(2)
        self.relu = nn.ReLU()
        self.seq_num = 0
        if virus_type == 'H1N1':
            self.seq_num = 82
        if virus_type == 'H3N2':
            self.seq_num = 83
        if virus_type == 'H5N1':
            self.seq_num = 81
        self.fc1 = nn.Linear(64 * self.seq_num * 26, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)
        self.dropout = nn.Dropout(p=dropout)
        self.logsoftmax = nn.LogSoftmax()

        self.contrastive_hidden_layer = nn.Linear(64, contrastive_dimension)
        self.contrastive_output_layer = nn.Linear(
            contrastive_dimension, contrastive_dimension)

    def freeze_projection(self):
        self.conv1.requires_grad_(False)
        self.bn1.requires_grad_(False)
        self.conv2.requires_grad_(False)
        self.bn2.requires_grad_(False)

    def _forward_impl_encoder(self, x):
        in_size = x.size(0)
        out = self.mp(self.conv1(x))
        out = self.bn1(out)
        out = self.mp(self.conv2(out))
        out = self.bn2(out)
        out = out.view(in_size, -1)
        out = self.relu(self.fc1(out))
        out = self.relu(self.fc2(out))
        out = F.normalize(out, dim=1)
        return out

    def forward_contrastive(self, x):
        # Implement from the encoder E to the projection network P
        x = self._forward_impl_encoder(x)
        x = self.contrastive_hidden_layer(x)
        x = F.relu(x)
        x = self.contrastive_output_layer(x)
        x = F.relu(x)
        # here, in the original code, they use normalize.
        x = F.normalize(x, dim=1)
        return x

    def forward(self, x):
        out = self._forward_impl_encoder(x)
        out = self.fc3(out)
        out = self.dropout(out)
        return self.logsoftmax(out)


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
        )

    def forward(self, x):
        w = F.adaptive_avg_pool2d(x, 1)  # Squeeze
        w = self.fc(x)
        w, b = w.split(w.data.size(1) // 2, dim=1)  # Excitation
        print(x.data.size())
        print(w.data.size())
        print(b.data.size())
        w = torch.sigmoid(w)

        return x * w + b  # Scale and add bias

# Residual Block with SEBlock


class ResBlock(nn.Module):
    def __init__(self, channels):
        super(ResBlock, self).__init__()

        self.conv_lower = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )

        self.conv_upper = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )

        self.se_block = SEBlock(channels)

    def forward(self, x):
        path = self.conv_lower(x)
        path = self.conv_upper(path)
        path = self.se_block(path)

        path = x + path
        return F.relu(path)

# Network Module


class IAV_CNN(nn.Module):
    def __init__(self, in_channel, filters, blocks, num_classes, contrastive_dimension=64, dropout=0.5):
        super(IAV_CNN, self).__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channel, filters, 3, padding=1, bias=False),
            nn.BatchNorm2d(filters),
            nn.ReLU()
        )

        self.res_blocks = nn.Sequential(
            *[ResBlock(filters) for _ in range(blocks - 1)])

        self.out_conv = nn.Sequential(
            nn.Conv2d(filters, 128, 1, padding=0, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.contrastive_hidden_layer = nn.Linear(128, contrastive_dimension)
        self.contrastive_output_layer = nn.Linear(contrastive_dimension, 128)

        self.fc = nn.Linear(128, num_classes)
        self.drop = nn.Dropout(p=dropout)

    def freeze_projection(self):
        self.conv_block.requires_grad_(False)
        self.res_blocks.requires_grad_(False)
        self.out_conv.requires_grad_(False)

    def _forward_impl_encoder(self, x):
        x = self.conv_block(x)
        x = self.res_blocks(x)
        x = self.out_conv(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.data.size(0), -1)
        return x

    def forward_contrastive(self, x):
        # Implement from the encoder E to the projection network P
        x = self._forward_impl_encoder(x)
        x = self.contrastive_hidden_layer(x)
        x = F.relu(x)
        x = self.contrastive_output_layer(x)
        # here, in the original code, they use normalize.
        return x

    def forward(self, x):
        x = self.conv_block(x)
        x = self.res_blocks(x)

        x = self.out_conv(x)
        # x = F.adaptive_avg_pool2d(x, 1)
        x = F.adaptive_max_pool2d(x, 1)
        x = x.view(x.data.size(0), -1)
        x = self.fc(x)
        x = self.drop(x)
        return F.log_softmax(x, dim=1)


class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

        # SE layers
        # Use nn.Conv2d instead of nn.Linear
        self.fc1 = nn.Conv2d(planes, planes//16, kernel_size=1)
        self.fc2 = nn.Conv2d(planes//16, planes, kernel_size=1)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        # Squeeze
        w = F.avg_pool2d(out, out.size(2))
        w = F.relu(self.fc1(w))
        w = F.sigmoid(self.fc2(w))
        # Excitation
        out = out * w  # New broadcasting feature from v0.2!

        out += self.shortcut(x)
        out = F.relu(out)
        return out


class PreActBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1,
                          stride=stride, bias=False)
            )

        # SE layers
        self.fc1 = nn.Conv2d(planes, planes//16, kernel_size=1)
        self.fc2 = nn.Conv2d(planes//16, planes, kernel_size=1)

    def forward(self, x):
        out = F.relu(x)
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(out))  # [32,64,325,100]

        # Squeeze
        w = F.avg_pool2d(out, [out.size(2), out.size(3)])  # [32,plane,1,1]

        w = F.relu(self.fc1(w))

        w = F.sigmoid(self.fc2(w))

        # Excitation
        out = out * w

        out += shortcut
        return out


class SENet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=2):
        super(SENet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(1, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block,  64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        #self.relu = nn.ReLU(100)
        self.linear = nn.Linear(15360, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.conv1(x))  # [32,64,325,100]
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)  # [bs,512,41,13]
        out = F.avg_pool2d(out, 4)  # [bs,512,10,3]
        out = out.view(out.size(0), -1)  # [bs,15360]
        #out = self.relu(out)
        out = self.linear(out)

        return out


def SENet18b():
    return SENet(BasicBlock, [3, 4, 6, 3])


def SENet18():
    return SENet(PreActBlock, [2, 2, 2, 2])
