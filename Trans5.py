"""
Transformer for EEG classification

The core idea is slicing, which means to split the signal along the time dimension. Slice is just like the patch in Vision Transformer.
"""

import mne
import os
import numpy as np
import math
import random
import time
import scipy.io

from torch.utils.data import DataLoader
from torch.autograd import Variable
#from torchsummary import summary

import torch
import torch.nn.functional as F

from torch import nn
from torch import Tensor

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from common_spatial_pattern import csp
import pandas as pdss
from sklearn.utils import shuffle
from itertools import combinations
# from confusion_matrix import plot_confusion_matrix
# from cm_no_normal import plot_confusion_matrix_nn
# from torchsummary import summary

#import matplotlib.pyplot as plt
# from torch.utils.tensorboard import SummaryWriter
# from torch.backends import cudnn
# cudnn.benchmark = False
# cudnn.deterministic = True

# writer = SummaryWriter('./TensorBoardX/')

# torch.cuda.set_device(6)
# gpus = [0]
# os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
# os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, gpus))

class PatchEmbedding(nn.Module):
    def __init__(self, emb_size):
        # self.patch_size = patch_size
        super().__init__()
        self.projection = nn.Sequential(
            nn.Conv2d(2, 2, (1, 51), (1, 1)),
            nn.BatchNorm2d(2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(2, emb_size, (16, 5), stride=(1, 5)),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        # self.positions = nn.Parameter(torch.randn((100 + 1, emb_size)))
        # self.positions = nn.Parameter(torch.randn((2200 + 1, emb_size)))

    def forward(self, x: Tensor) -> Tensor:
        b, _, _, _ = x.shape
        x = self.projection(x)
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)

        # position
        # x += self.positions
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)  # batch, num_heads, query_len, key_len
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )

class GELU(nn.Module):
    def forward(self, input: Tensor) -> Tensor:
        return input*0.5*(1.0+torch.erf(input/math.sqrt(2.0)))


class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size,
                 num_heads=5,
                 drop_p=0.5,
                 forward_expansion=4,
                 forward_drop_p=0.5):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, num_heads, drop_p),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
            ))


class TransformerEncoder(nn.Sequential):
    def __init__(self, depth, emb_size):
        super().__init__(*[TransformerEncoderBlock(emb_size) for _ in range(depth)])


class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size, n_classes):
        super().__init__()
        self.clshead = nn.Sequential(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, n_classes)
        )

    def forward(self, x):
        out = self.clshead(x)
        return x, out


class ViT(nn.Sequential):
    def __init__(self, emb_size=10, depth=3, n_classes=2, **kwargs):
    #def __init__(self, emb_size=5, depth=2, n_classes=2, **kwargs):

        super().__init__(
            # channel_attention(),
            ResidualAdd(
                nn.Sequential(
                    nn.LayerNorm(3000),
                    channel_attention(),
                    nn.Dropout(0.5),
                )
            ),

            PatchEmbedding(emb_size),
            TransformerEncoder(depth, emb_size),
            ClassificationHead(emb_size, n_classes)
        )


class channel_attention(nn.Module):
    def __init__(self, sequence_num=3000, inter=30):
        super(channel_attention, self).__init__()
        self.sequence_num = sequence_num
        self.inter = inter
        self.extract_sequence = int(self.sequence_num / self.inter)  # You could choose to do that for less computation

        self.query = nn.Sequential(
            nn.Linear(16, 16), #32*32
            nn.LayerNorm(16),  # also may introduce improvement to a certain extent
            nn.Dropout(0.3)
        )
        self.key = nn.Sequential(
            nn.Linear(16, 16),
            # result2
            #nn.LeakyReLU(),

            nn.LayerNorm(16),
            nn.Dropout(0.3)
        )

        # self.value = self.key
        self.projection = nn.Sequential(
            nn.Linear(16, 16),
            #result2
            #nn.LeakyReLU(),

            nn.LayerNorm(16),
            nn.Dropout(0.3),
        )

        self.drop_out = nn.Dropout(0)
        self.pooling = nn.AvgPool2d(kernel_size=(1, self.inter), stride=(1, self.inter))

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        temp = rearrange(x, 'b o c s->b o s c')
        temp_query = rearrange(self.query(temp), 'b o s c -> b o c s')
        temp_key = rearrange(self.key(temp), 'b o s c -> b o c s')

        channel_query = self.pooling(temp_query)
        channel_key = self.pooling(temp_key)

        scaling = self.extract_sequence ** (1 / 2)

        channel_atten = torch.einsum('b o c s, b o m s -> b o c m', channel_query, channel_key) / scaling

        channel_atten_score = F.softmax(channel_atten, dim=-1)
        channel_atten_score = self.drop_out(channel_atten_score)

        out = torch.einsum('b o c s, b o c m -> b o c s', x, channel_atten_score)
        '''
        projections after or before multiplying with attention score are almost the same.
        '''
        out = rearrange(out, 'b o c s -> b o s c')
        out = self.projection(out)
        out = rearrange(out, 'b o s c -> b o c s')
        return out


class Trans():
    def __init__(self, nsub: int):
        super(Trans, self).__init__()
        self.batch_size = 10 
        self.n_epochs = 1000
        self.img_height = 22
        self.img_width = 600
        self.channels = 22 # EEG channels 22      # 1
        self.c_dim = 4
        self.lr = 0.0002
        self.b1 = 0.5
        self.b2 = 0.9
        self.nSub = nsub
        self.start_epoch = 0
        self.root = ''  # the path of data

        self.pretrain = False

        self.log_write = open("./results2/log_subject%d.txt" % self.nSub, "w")

        self.img_shape = (self.channels, self.img_height, self.img_width)  # something no use

        self.Tensor = torch.FloatTensor
        self.LongTensor = torch.LongTensor

        self.criterion_l1 = torch.nn.L1Loss()
        self.criterion_l2 = torch.nn.MSELoss()
        self.criterion_cls = torch.nn.CrossEntropyLoss()

        self.model = ViT()
        if self.pretrain:
            self.model.load_state_dict(torch.load(f'./models/model_%d.pth' % self.nSub))

        # self.model = nn.DataParallel(self.model, device_ids=[i for i in range(len(gpus))])
        # self.model = self.model
        #summary(self.model, (1, 16, 1000), device='cpu')

        self.centers = {}
   
    def get_source_data(self):

        # to get the data of target subject
        self.total_data = scipy.io.loadmat(self.root + 'data/mat/A0%dT.mat' % self.nSub)
        self.train_data = self.total_data['data']
        self.train_label = self.total_data['label']
        

        self.train_data = np.transpose(self.train_data, (2, 1, 0))
        self.train_data = np.expand_dims(self.train_data, axis=1)
        self.train_label = np.transpose(self.train_label)

        self.allData = self.train_data
        self.allLabel = self.train_label[0]

        # test data
        # to get the data of target subject
        self.test_tmp = scipy.io.loadmat(self.root + 'data/mat/A0%dE.mat' % self.nSub)
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
    
    def normalize_vector(self,vector):
        normalized_vector = (vector - vector.min()) / (vector.max() - vector.min())
        return normalized_vector
    
    def preprocess(self,X):
        X = np.array(X)
        X = np.array([self.normalize_vector(i) for i in X], dtype=np.float)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        return X
    

    def remove_file(self,file_loc):
        if os.path.exists(file_loc):
            os.remove(file_loc)
            print("File removed")
        else:
            print("The file does not exist")

    
    def build_dataset(self,no_of_people):
        dataset = {}
        X = []
        Y = []
        folders = os.listdir(os.getcwd() + '/files')
        for folder in folders[:no_of_people]:
            y = int(folder[-3:])-1
            count = 0
            for filename in os.listdir(os.getcwd()+ '/files'+'/'+folder):
                task_no = filename[5:7]
                folder_no = folder[1:]
                if filename.endswith("edf") and task_no != "01" and task_no != "02":
                    edf_file = mne.io.read_raw_edf(os.getcwd()+'/files'+'/'+folder+'/'+filename)
                    eeg = edf_file.get_data()
                    if eeg.shape[1] > 15000:
                        eeg = np.moveaxis(np.asarray(eeg, dtype=np.float64), 0, -1)
                        data = eeg[:3000, :]
                        data = np.array([self.normalize_vector(i) for i in data], dtype=np.float)
                        X.append(data)
                        Y.append(y)
                else:
                    continue
        
        dataset['x'] = np.array(X)
        dataset['y'] = np.array(Y) 
        #print(dataset["x"].shape, dataset["y"].shape)#(24, 3000, 64) (24,) s001 s002
        return dataset
    

    def split_data(self,dataset, val_split = 0.0, test_split = 0.2, channel_index_start = 0, channel_index_end = 1):
        x_train = []
        y_train = []
        x_val = []
        y_val = []
        x_test = []
        y_test = []

        #print(dataset["x"].shape, dataset["y"].shape)
        x = dataset["x"][:,:,channel_index_start:channel_index_end]
        y = dataset["y"]
        no_of_classes = len(np.unique(y))
        for i in range(no_of_classes):
            subject_idx = np.where(y == i)
            idx = subject_idx[0]

            val_count = int(val_split*len(idx))
            test_count = int(test_split*len(idx))
            
            val_idx = np.random.choice(idx, size=val_count, replace=False)
            for k in val_idx:
                idx = np.delete(idx, np.argwhere(idx == k))
            
            test_idx = np.random.choice(idx, size=test_count, replace=False)
            for k in test_idx:
                idx = np.delete(idx, np.argwhere(idx == k))

            for j in val_idx:
                x_val.append(x[j])
                y_val.append(y[j])
            
            for j in test_idx:
                x_test.append(x[j])
                y_test.append(y[j])
            
            for j in idx:
                x_train.append(x[j])
                y_train.append(y[j])
                
        x_train, y_train = np.array(x_train), np.array(y_train)
        x_val, y_val = np.array(x_val), np.array(y_val)
        x_test, y_test = np.array(x_test), np.array(y_test)

        x_train, y_train = shuffle(x_train, y_train)
        x_val, y_val = shuffle(x_val, y_val)
        x_test, y_test = shuffle(x_test, y_test)

        return (x_train, y_train, x_test, y_test)
    

    def make_pairs(self,data, labels):
        pair_signals = []
        pair_subjects = []

        no_of_classes = len(np.unique(labels))
        idx = [np.where(labels == i)[0] for i in range(0, no_of_classes)]

        for person in idx:
            positive_combinations = combinations(person, 2)
            for pair in positive_combinations:
                signal_1_idx = pair[0]
                signal_2_idx = pair[1] 
                current_signal = data[signal_1_idx]
                subject = labels[signal_1_idx]
                pos_signal = data[signal_2_idx]
                pair_signals.append([current_signal, pos_signal])
                # pair_subjects.append((1.0, subject))
                pair_subjects.append(1.0)

                neg_idx = np.where(labels != subject)[0]
                neg_signal = data[np.random.choice(neg_idx)]
                pair_signals.append([current_signal, neg_signal])
                # pair_subjects.append((0.0, subject))
                pair_subjects.append(0.0)#False
        
        # for idxA in range(len(data)):
        #   current_signal = data[idxA]
        #   subject = labels[idxA]
        #   idxB = np.random.choice(idx[subject])
        #   pos_signal = data[idxB]
        #   pair_signals.append([current_signal, pos_signal])
        #   pair_subjects.append(([1.0], subject))

        #   neg_idx = np.where(labels != subject)[0]
        #   neg_signal = data[np.random.choice(neg_idx)]
        #   pair_signals.append([current_signal, neg_signal])
        #   pair_subjects.append(([0.0], subject))

        return (np.array(pair_signals), np.array(pair_subjects))
    
    def update_lr(self, optimizer, lr):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    # Do some data augmentation is a potential way to improve the generalization ability
    def aug(self, img, label):
        aug_data = []
        aug_label = []
        return aug_data, aug_label

    def train(self):

        # file_remove_list = os.listdir(os.getcwd() + '\\files')
        # file_remove_list = ["64_channel_sharbrough-old.png", "64_channel_sharbrough.pdf", "64_channel_sharbrough.png", "ANNOTATORS", "RECORDS", "SHA256SUMS.txt", "eeg-motor-movementimagery-dataset-1.0.0.zip","wfdbcal"]
        # for files in file_remove_list:
        #     self.remove_file(files)
        dataset = self.build_dataset(109) #no_of_people=109 (24, 3000, 64) (24,) 
        x_train, y_train, x_test, y_test = self.split_data(dataset, 0.0, 0.2, 7, 23)

        #print("x_train shape", x_train.shape) #(20, 3000, 16)
        (img, label) = self.make_pairs(x_train, y_train)
        (test_data, test_label) = self.make_pairs(x_test, y_test)

        img = torch.from_numpy(img)  #[180, 2, 3000, 16] [288,1,16,1000] 288
        label = torch.from_numpy(label) #[180]
        # put img's last dimension to the second
        img = img.permute(0,1,3,2 ) #[180, 16, 2, 3000]
        # print("img shape", img.shape)
        # print("label shape", label.shape)
        dataset = torch.utils.data.TensorDataset(img, label)
        self.dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=True)

        test_data = torch.from_numpy(test_data)
        test_label = torch.from_numpy(test_label)
        test_data = test_data.permute(0, 1,3, 2)
        test_dataset = torch.utils.data.TensorDataset(test_data, test_label)
        self.test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=self.batch_size, shuffle=True)
        # #print("img shape", img.shape)
        # label = torch.from_numpy(label)
        #print("label shape", label.shape)
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

        print('Training...')
        # print(img.shape) #[180, 2, 3000, 1]
        # print(label.shape) [180]
        for e in range(self.n_epochs):
            in_epoch = time.time()
            self.model.train()
            for i, (img, label) in enumerate(self.dataloader):
                img = Variable(img.type(self.Tensor))
                label = Variable(label.type(self.LongTensor))
                tok, outputs = self.model(img)
                #(outputs.shape)

                loss = self.criterion_cls(outputs, label)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            out_epoch = time.time()
            
            #TEST
            if (e + 1) % 1 == 0:
                self.model.eval()
                Tok, Cls = self.model(test_data)
                #print(Cls)
                #print(test_label)
                loss_test = self.criterion_cls(Cls, test_label)
                y_pred = torch.max(Cls, 1)[1]
                #print(y_pred)
                acc = float((y_pred == test_label).cpu().numpy().astype(int).sum()) / float(test_label.size(0))
                train_pred = torch.max(outputs, 1)[1]
                train_acc = float((train_pred == label).cpu().numpy().astype(int).sum()) / float(label.size(0))
                print('Epoch:', e,
                      '  Train loss:', loss.detach().cpu().numpy(),
                      '  Test loss:', loss_test.detach().cpu().numpy(),
                      '  Train accuracy:', train_acc,
                      '  Test accuracy is:', acc,
                      '  Train_predict:', train_pred,
                      '  label:', label)
                self.log_write.write(str(e) + "    " + str(acc) + "\n")
                num = num + 1
                averAcc = averAcc + acc
                if acc > bestAcc:
                    bestAcc = acc
                    Y_true = test_label
                    Y_pred = y_pred

        torch.save(self.model.state_dict(), f'models2/model_{self.nSub}.pth')
        averAcc = averAcc / num
        print('The average accuracy is:', averAcc)
        print('The best accuracy is:', bestAcc)
        self.log_write.write('The average accuracy is: ' + str(averAcc) + "\n")
        self.log_write.write('The best accuracy is: ' + str(bestAcc) + "\n")

        return bestAcc, averAcc, Y_true, Y_pred


def main():
    best = 0
    aver = 0
    result_write = open("./results2/sub_result.txt", "w")
    i = 0
    #for i in range(9):
    seed_n = np.random.randint(500)
    print('seed is ' + str(seed_n))
    random.seed(seed_n)
    np.random.seed(seed_n)
    torch.manual_seed(seed_n)
    torch.manual_seed(seed_n)
    torch.manual_seed(seed_n)
    print('Subject %d' % (i+1))
    trans = Trans(i + 1)
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


    #best = best / 9
    #aver = aver / 9
    # plot_confusion_matrix(yt, yp, 666)
    result_write.write('**The average Best accuracy is: ' + str(best) + "\n")
    result_write.write('The average Aver accuracy is: ' + str(aver) + "\n")
    result_write.close()


if __name__ == "__main__":
    main()
