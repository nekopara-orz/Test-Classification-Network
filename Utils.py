
from thop import profile
from thop import clever_format


import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F

import Loader


def count_param(model):
    param_count = 0
    for param in model.parameters():
        param_count += param.view(-1).size()[0]
    return param_count


def count_flops_param(model, inputs=None):
    macs, params = profile(model, (inputs,))
    macs, params = clever_format([macs, params], "%.3f")
    return macs, params


def test_model(epoch=20, model=None, size=(224, 224)):
    train_loader, test_loader = Loader.load(size=size)

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    for e in range(epoch):
        model.train()

        train_loss = 0.0
        criterion = nn.CrossEntropyLoss()

        # train_loader.set_description('[%s%04d/%04d %s%f]' % ('Epoch:', e + 1, epoch, 'lr:', 0.001))

        for i, data in enumerate(train_loader, 0):  # 0是下标起始位置默认为0

            start = datetime.datetime.now()  # 计时
            inputs, labels = data[0], data[1]
            # 初始为0，清除上个batch的梯度信息
            optimizer.zero_grad()
            outputs = model(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # topk 准确率计算

            # ans = torch.argsort(outputs, 1)
            _, ans = torch.topk(outputs, 1, 1, True, True)
            ans = ans.t()
            print(ans)
            correct = ans.eq(labels.view(1, -1))
            print(correct)
            print(correct.view(-1))
            correct_k = correct.view(-1).float().sum(0)

            print(correct_k.item(), len(labels), "acc = {:.3f}".format(correct_k.item() / len(labels)))

            end = datetime.datetime.now()
            print('time :', (end - start).total_seconds())
