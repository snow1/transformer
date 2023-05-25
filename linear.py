import os
import numpy as np
import math
import random
import time
import scipy.io

from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchsummary import summary

import torch
import torch.nn.functional as F

from torch import nn
from torch import Tensor

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from common_spatial_pattern import csp


class linear():
    def __init__(self, nsub):
        super(linear, self).__init__()
        self.batch_size = 50
        self.n_epochs = 1000
        self.img_height = 22
        self.img_width = 600
        self.channels = 1
        self.c_dim = 4
        self.lr = 0.0002
        self.b1 = 0.5
        self.b2 = 0.9
        self.nSub = nsub
        self.start_epoch = 0
        self.root = ''  # the path of data

        self.pretrain = False

        self.log_write = open("./linearResults/log_subject%d.txt" % self.nSub, "w")

        self.img_shape = (self.channels, self.img_height, self.img_width)  # something no use

        self.Tensor = torch.FloatTensor
        self.LongTensor = torch.LongTensor

        self.criterion_l1 = torch.nn.L1Loss()
        self.criterion_l2 = torch.nn.MSELoss()
        self.criterion_cls = torch.nn.CrossEntropyLoss()

        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16000, 100),
            nn.ReLU(),
            nn.Linear(100, 4),
            nn.Softmax(dim=1)
        )
        # self.model = nn.DataParallel(self.model, device_ids=[i for i in range(len(gpus))])
        # self.model = self.model
       # summary(self.model, (16,16), device='cpu')

        self.centers = {}

    def get_source_data(self):

        # to get the data of target subject
        self.total_data = scipy.io.loadmat(self.root + 'A0%dT.mat' % self.nSub)
        self.train_data = self.total_data['data']
        self.train_label = self.total_data['label']

        self.train_data = np.transpose(self.train_data, (2, 1, 0))
        self.train_data = np.expand_dims(self.train_data, axis=1)
        self.train_label = np.transpose(self.train_label)

        self.allData = self.train_data
        self.allLabel = self.train_label[0]

        # test data
        # to get the data of target subject
        self.test_tmp = scipy.io.loadmat(self.root + 'A0%dE.mat' % self.nSub)
        self.test_data = self.test_tmp['data']
        self.test_label = self.test_tmp['label']

        # self.train_data = self.train_data[250:1000, :, :]
        self.test_data = np.transpose(self.test_data, (2, 1, 0))
        self.test_data = np.expand_dims(self.test_data, axis=1)
        self.test_label = np.transpose(self.test_label)

        self.testData = self.test_data
        self.testLabel = self.test_label[0]

        # standardize
        target_mean = np.mean(self.allData)
        target_std = np.std(self.allData)
        self.allData = (self.allData - target_mean) / target_std
        self.testData = (self.testData - target_mean) / target_std

        tmp_alldata = np.transpose(np.squeeze(self.allData), (0, 2, 1))
        Wb = csp(tmp_alldata, self.allLabel-1)  # common spatial pattern
        self.allData = np.einsum('abcd, ce -> abed', self.allData, Wb)
        self.testData = np.einsum('abcd, ce -> abed', self.testData, Wb)
        return self.allData, self.allLabel, self.testData, self.testLabel

    def update_lr(self, optimizer, lr):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    # Do some data augmentation is a potential way to improve the generalization ability
    def aug(self, img, label):
        aug_data = []
        aug_label = []
        return aug_data, aug_label

    def train(self):


        img, label, test_data, test_label = self.get_source_data()
        img = torch.from_numpy(img)
        label = torch.from_numpy(label - 1)


        dataset = torch.utils.data.TensorDataset(img, label)
        self.dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=True)

        test_data = torch.from_numpy(test_data)
        test_label = torch.from_numpy(test_label - 1)
        test_dataset = torch.utils.data.TensorDataset(test_data, test_label)
        self.test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=self.batch_size, shuffle=True)

        # Optimizers
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(self.b1, self.b2))

        test_data = Variable(test_data.type(self.Tensor))
        test_label = Variable(test_label.type(self.LongTensor))

        bestAcc = 0
        averAcc = 0
        num = 0
        Y_true = 0
        Y_pred = 0

        # Train the cnn model
        total_step = len(self.dataloader)
        curr_lr = self.lr
        # some better optimization strategy is worthy to explore. Sometimes terrible over-fitting.


        for e in range(self.n_epochs):
            in_epoch = time.time()
            self.model.train()
            for i, (img, label) in enumerate(self.dataloader):

                img = Variable(img.type(self.Tensor))
                label = Variable(label.type(self.LongTensor))
                outputs = self.model(img)               
               # print(outputs.shape)
                loss = self.criterion_cls(outputs, label)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            out_epoch = time.time()

            if (e + 1) % 1 == 0:
                self.model.eval()
                Cls = self.model(test_data)

                loss_test = self.criterion_cls(Cls, test_label)
                y_pred = torch.max(Cls, 1)[1]
                acc = float((y_pred == test_label).cpu().numpy().astype(int).sum()) / float(test_label.size(0))
                train_pred = torch.max(outputs, 1)[1]
                train_acc = float((train_pred == label).cpu().numpy().astype(int).sum()) / float(label.size(0))
                print('Epoch:', e,
                      '  Train loss:', loss.detach().cpu().numpy(),
                      '  Test loss:', loss_test.detach().cpu().numpy(),
                      '  Train accuracy:', train_acc,
                      '  Test accuracy is:', acc)
                self.log_write.write(str(e) + "    " + str(acc) + "\n")
                num = num + 1
                averAcc = averAcc + acc
                if acc > bestAcc:
                    bestAcc = acc
                    Y_true = test_label
                    Y_pred = y_pred

        torch.save(self.model.state_dict(), f'linearModels/model_{self.nSub}.pth')
        averAcc = averAcc / num
        print('The average accuracy is:', averAcc)
        print('The best accuracy is:', bestAcc)
        self.log_write.write('The average accuracy is: ' + str(averAcc) + "\n")
        self.log_write.write('The best accuracy is: ' + str(bestAcc) + "\n")

        return bestAcc, averAcc, Y_true, Y_pred

def main():
    best = 0
    aver = 0
    result_write = open("./linearResults/sub_result.txt", "w")

    for i in range(9):
        seed_n = np.random.randint(500)
        print('seed is ' + str(seed_n))
        random.seed(seed_n)
        np.random.seed(seed_n)
        torch.manual_seed(seed_n)
        print('Subject %d' % (i+1))
        trans = linear(i + 1)
        bestAcc, averAcc, Y_true, Y_pred = trans.train()
        print('THE BEST ACCURACY IS ' + str(bestAcc))
        result_write.write('Subject ' + str(i + 1) + ' : ' + 'Seed is: ' + str(seed_n) + "\n")
        result_write.write('**Subject ' + str(i + 1) + ' : ' + 'The best accuracy is: ' + str(bestAcc) + "\n")
        result_write.write('Subject ' + str(i + 1) + ' : ' + 'The average accuracy is: ' + str(averAcc) + "\n")
        # plot_confusion_matrix(Y_true, Y_pred, i+1)
        best = best + bestAcc
        aver = aver + averAcc
        if i == 0:
            yt = Y_true
            yp = Y_pred
        else:
            yt = torch.cat((yt, Y_true))
            yp = torch.cat((yp, Y_pred))


    best = best / 9
    aver = aver / 9
    # plot_confusion_matrix(yt, yp, 666)
    result_write.write('**The average Best accuracy is: ' + str(best) + "\n")
    result_write.write('The average Aver accuracy is: ' + str(aver) + "\n")
    result_write.close()



if __name__ == "__main__":
    main()
