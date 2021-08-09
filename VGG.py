import datetime

import torch
import torch.nn as nn
import math

import Loader
import Utils

from Loader import MyDataset
from Loader import train_path
from Loader import test_path

class VGG(nn.Module):

    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def vgg11(**kwargs):
    model = VGG(make_layers(cfg['A']), **kwargs)
    return model


def vgg11_bn(**kwargs):
    model = VGG(make_layers(cfg['A'], batch_norm=True), **kwargs)
    return model


def vgg13(**kwargs):
    model = VGG(make_layers(cfg['B']), **kwargs)
    return model


def vgg13_bn(**kwargs):
    model = VGG(make_layers(cfg['B'], batch_norm=True), **kwargs)
    return model


def vgg16(**kwargs):
    model = VGG(make_layers(cfg['D']), **kwargs)
    return model


def vgg16_bn(**kwargs):
    model = VGG(make_layers(cfg['D'], batch_norm=True), **kwargs)
    return model


def vgg19(**kwargs):
    model = VGG(make_layers(cfg['E']), **kwargs)
    return model


def vgg19_bn(**kwargs):
    model = VGG(make_layers(cfg['E'], batch_norm=True), **kwargs)
    return model


def vgg16test(epoch=20):

    train_ds = MyDataset(train_path)
    test_ds = MyDataset(test_path)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=32,
                                               shuffle=True, pin_memory=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=32,
                                              shuffle=True, pin_memory=True, num_workers=0)

    model = vgg16(num_classes=2)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    for e in range(epoch):
        model.train()

        train_loss = 0.0
        criterion = nn.CrossEntropyLoss()

        # train_loader.set_description('[%s%04d/%04d %s%f]' % ('Epoch:', e + 1, epoch, 'lr:', 0.001))

        for i, data in enumerate(train_loader, 0):  # 0是下标起始位置默认为0

            start = datetime.datetime.now()  #计时
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
            correct = ans.eq( labels.view(1, -1))
            print(correct)
            print(correct.view(-1))
            correct_k = correct.view(-1).float().sum(0)

            print(correct_k.item(), len(labels),"acc = {:.3f}".format(correct_k.item() / len(labels)))

            end = datetime.datetime.now()
            print('time :', (end - start).total_seconds())

            # prec1, prec2 = accuracy(outputs, labels, topk=(1, 2))
            # n = inputs.size(0)
            # top1.update(prec1.item(), n)
            # train_loss += loss.item()
            # postfix = {'train_loss': '%.6f' % (train_loss / (i + 1)), 'train_acc': '%.6f' % top1.avg}
            # train_loader.set_postfix(log=postfix)

            # ternsorboard 曲线绘制
            # writer = SummaryWriter(tensorboard_path)
            # writer.add_scalar('Train/Loss', loss.item(), epoch)
            # writer.add_scalar('Train/Accuracy', top1.avg, epoch)
            # writer.flush()


def vgg19test(epoch=20):
    train_ds = MyDataset(train_path)
    test_ds = MyDataset(test_path)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=32,
                                               shuffle=True, pin_memory=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=32,
                                              shuffle=True, pin_memory=True, num_workers=0)

    model = vgg19(num_classes=2)
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


if __name__ == '__main__':
    # 'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19_bn', 'vgg19'
    # Example
    # net16 = vgg16()
    # net19 = vgg19()
    # p16 = Utils.count_param(net16)
    # p19 = Utils.count_param(net19)
    # ins = torch.randn([1, 3, 224, 224], dtype=torch.float32)
    # print(ins)
    # flops16, param16 = Utils.count_flops_param(net16, ins)
    # flops19, param19 = Utils.count_flops_param(net19, ins)
    #
    # print("vgg16 flops= {0},vgg16 param = {1} ".format(flops16, param16))
    # print("vgg19 flops= {0},vgg19 param = {1} ".format(flops19, param19))
    # print("param vgg16 = {0},param vgg19 = {1} ".format(p16, p19))
    # print(net16)

    # 测试VGG16在猫狗数据集上运行
    # vgg16test(epoch=20)

    # 测试VGG19在猫狗数据集上运行
    vgg19test(epoch=20)


