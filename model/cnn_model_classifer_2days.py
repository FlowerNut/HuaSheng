from torch import nn, optim
import torch
from torch.utils.data import random_split, DataLoader
import datetime
import numpy as np
from model.train_data_iterator import DataFormer
import os
from model import train_config

# --------------------------conv2:--------------------------------------
# 如果想要 con2d 出来的图片长宽没有变化, padding=(kernel_size-1)/2 当 stride=1
# out_size = （in_size - kernel_size + 2*padding）/ stride +1
# for example: pic=5x5, kernel=4x4, stride=1, padding=0, output=3
# --------------------------pooling:--------------------------------------
# output_size = (in_size - filter_size)/stride +1


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(1, 6, 3, stride=1, padding=1),   # 6层滤器， 公式（5-3+2*1）/1+1=5
            nn.ReLU(True),
            nn.AvgPool3d(3, stride=1),  # (5-3)/1+1=3
            nn.Conv3d(6, 12, 2, stride=1, padding=1),  # 3层滤器, (3-2+2*1)/1+1=4
            nn.ReLU(True),
            nn.AvgPool3d(3, stride=1)  # (4-3)/1+1=2, 共2x2x3=12个元素
        )
        self.fc = nn.Sequential(
            # 输入12xstream_len，输出2
            nn.Linear(336, 84),
            nn.ReLU(),
            nn.Linear(84, 20),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.conv(x)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out


class Classifier:
    def __init__(self, model_file_name):
        self.model_file_path = os.path.join(train_config.model_directory_path, model_file_name)
        self.data_split = 0.7
        self.learning_rate = 1e-3
        self.epoch = 20
        self.batch_size = 100
        self.stream_len = 10  # 使用两周fft数据进行预测
        self.__trainer()

    # 保存日志
    # fname是要保存的位置，s是要保存的内容
    def __log(self, filename, s):
        f = open(filename, 'a')
        f.write(str(datetime.datetime.now()) + ': ' + s + '\n')
        f.close()

    def __data_former(self):
        dataset = DataFormer(self.stream_len)
        train_dataset_len = int(len(dataset)*self.data_split)
        valid_dataset_len = len(dataset) - train_dataset_len
        train_dataset, valid_dataset = random_split(
            dataset=dataset,
            lengths=[train_dataset_len, valid_dataset_len],
            generator=torch.Generator().manual_seed(0)
        )
        return train_dataset, valid_dataset

    def __trainer(self):
        # data load
        train_dataset, valid_dataset = self.__data_former()
        train_data_loader = DataLoader(
            dataset=train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2
        )
        valid_data_loader = DataLoader(
            dataset=valid_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2
        )
        # net init
        cnn = CNN()
        optimizer = optim.Adam(cnn.parameters(), lr=self.learning_rate)
        loss_func = nn.CrossEntropyLoss()  # 分类问题
        # 定义学习率衰减点，训练到50%和75%时学习率缩小为原来的1/10
        mult_step_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[self.epoch // 2, self.epoch // 4 * 3],
                                                             gamma=0.1)
        # 训练+验证
        progress_train_loss = []
        progress_valid_loss = []
        progress_valid_accuracy = []
        min_valid_loss = np.inf
        for i in range(self.epoch):
            print("epoch:{0}".format(i))
            # =============================训练=============================
            total_train_loss = []
            cnn.train()  # 进入训练模式
            for step, (image, label) in enumerate(train_data_loader):
                #         lr = set_lr(optimizer, i, EPOCH, LR)
                #image = image.type(torch.FloatTensor)  #
                #label = label.type(torch.long)  # CrossEntropy的target是longtensor，且要是1-D，不是one hot编码形式
                prediction = cnn(image)  # cnn output
                prediction = prediction.view(1, -1)  # 预测结果拉平成一行
                label = label.view(1, -1) #.type(torch.FloatTensor)
                #prediction = prediction[prediction.size(1)-label.size(1):].type(torch.FloatTensor)
                loss = loss_func(prediction, label)  # 计算损失
                optimizer.zero_grad()  # clear gradients for this training step
                loss.backward()  # backpropagation, compute gradients
                optimizer.step()  # apply gradients
                total_train_loss.append(loss.item())
            progress_train_loss.append(np.mean(total_train_loss))  # 存入平均交叉熵
            # ===============================测试============================
            total_valid_loss = []
            valid_accuracy = []
            cnn.eval()
            for step, (image, label) in enumerate(valid_data_loader):
                image = image.type(torch.FloatTensor).to()
                label = label.type(torch.FloatTensor).to()
                with torch.no_grad():
                    prediction = cnn(image)  # rnn output
                    prediction = prediction.view(1, -1)
                    label = label.view(1, -1) #.type(torch.FloatTensor)
                    #prediction = prediction[prediction.size(1) - label.size(1):].type(torch.FloatTensor)
                #         h_s = h_s.data        # repack the hidden state, break the connection from last iteration
                #         h_c = h_c.data        # repack the hidden state, break the connection from last iteration
                loss = loss_func(prediction, label)  # calculate loss
                total_valid_loss.append(loss.item())
                valid_accuracy.append(self.__accuracy_cal(prediction, label))
            progress_valid_loss.append(np.mean(total_valid_loss))
            progress_valid_accuracy.append(np.mean(valid_accuracy))

            if progress_valid_loss[-1] < min_valid_loss:
                '''
                torch.save({'epoch': i, 'model': cnn, 'train_loss': progress_train_loss,
                            'valid_loss': progress_valid_loss, 'valid_accuracy': progress_valid_accuracy},
                           './CNN.model')  # 保存字典对象，里面'model'的value是模型
                #         torch.save(optimizer, './CNN.optim')     # 保存优化器
                '''
                torch.save(cnn, self.model_file_path)    # './cnn_net.pth'
                min_valid_loss = progress_valid_loss[-1]

            # 编写日志
            log_string = ('iter: [{:d}/{:d}], train_loss: {:0.6f}, valid_loss: {:0.6f}, valid_accuracy:{:0.6f},'
                          'best_valid_loss: {:0.6f}, lr: {:0.7f}').format((i + 1), self.epoch,
                                                                          progress_train_loss[-1],
                                                                          progress_valid_loss[-1],
                                                                          progress_valid_accuracy[-1],
                                                                          min_valid_loss,
                                                                          optimizer.param_groups[0]['lr'])
            mult_step_scheduler.step()  # 学习率更新
            # 服务器一般用的世界时，需要加8个小时，可以视情况把加8小时去掉
            print(str(datetime.datetime.now() + datetime.timedelta(hours=8)) + ': ')
            print(log_string)  # 打印日志
            self.__log('./CNN.log', log_string)  # 保存日志

    def __accuracy_cal(self, prediction: torch.tensor, label: torch.tensor) -> float:
        p = prediction.view(1, -1)   #拉直成一行
        l = label.view(1, -1)
        res = torch.sum(torch.abs((l - p)))
        res = res.item()
        return res
