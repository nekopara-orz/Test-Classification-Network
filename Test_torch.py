import numpy as np
import torch
from torch.utils import data  # 获取迭代数据
from torch.autograd import Variable  # 获取变量
from thop import profile

PATH = 'model1.pth'


class MyCnn(torch.nn.Module):

    def __init__(self):
        super(MyCnn, self).__init__()
        self.h1 = torch.nn.Linear(1, 100)
        self.h2 = torch.nn.Linear(100, 1)

    def forward(self, x):
        x = self.h1(x)
        x = torch.relu(x)
        x = self.h2(x)
        return x


if __name__ == '__main__':

    net = MyCnn()
    loss_f = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
    ins = torch.tensor([x / 100 for x in range(100)])
    inputs = torch.unsqueeze(ins, 1)

    ts = torch.tensor([x * x for x in inputs])
    targets = torch.unsqueeze(ts, 1)

    flops, params = profile(net, (torch.tensor([[1.0]]),))
    print("flops = {0}, params = {1}".format(flops, params))

    for i in range(1000):
        outs = net(inputs)
        loss = loss_f(outs, targets)
        # print(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    outs = net(inputs)

    # torch.save(net.state_dict(), PATH)
    torch.save(net, PATH)
    m = torch.load(PATH)
    m.eval()
    l = m(inputs)

    flops, params = profile(m, (inputs,))
    print("flops = {0}, params = {1}".format(flops, params))

    # for i in range(100):
    #     print("in {0} tar {1} out{2}".format(ins[i], ts[i], outs[i]))
