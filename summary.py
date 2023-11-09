# --------------------------------------------#
#   该部分代码用于看网络结构
# --------------------------------------------#
import torch
from thop import clever_format, profile

from utils.options import args_parser
from pytorchyolo.models import load_model

import time

if __name__ == "__main__":
    input_shape = [416, 416]

    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    m = load_model(args.config, args.device, args.pretrained_weights)
    for i in m.children():
        print(i)
        print('==============================')

    dummy_input = torch.randn(1, 3, input_shape[0], input_shape[1]).to(args.device)
    flops, params = profile(m.to(args.device), (dummy_input,), verbose=False)
    # --------------------------------------------------------#
    #   flops * 2是因为profile没有将卷积作为两个operations
    #   有些论文将卷积算乘法、加法两个operations。此时乘2
    #   有些论文只考虑乘法的运算次数，忽略加法。此时不乘2
    #   本代码选择乘2，参考YOLOX。
    # --------------------------------------------------------#
    flops = flops * 2
    flops, params = clever_format([flops, params], "%.3f")
    print('Total GFLOPS: %s' % (flops))
    print('Total params: %s' % (params))

    torch.cuda.synchronize()
    start = time.time()

    result = m(dummy_input)

    torch.cuda.synchronize()
    end = time.time()

    print('infer time: ', end-start)