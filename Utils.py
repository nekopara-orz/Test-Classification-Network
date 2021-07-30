from thop import profile
from thop import clever_format

def count_param(model):
    param_count = 0
    for param in model.parameters():
        param_count += param.view(-1).size()[0]
    return param_count


def count_flops_param(model, inputs=None):
    macs, params = profile(model, (inputs,))
    macs, params = clever_format([macs, params], "%.3f")
    return macs, params
