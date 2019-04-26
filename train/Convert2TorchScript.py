# -*- coding: utf-8 -*-
import sys
sys.path.append('./')
import os

import argparse
import torch
import models

def main(config):
    # create directories if not exist
    if not os.path.exists(config.out_path):
        os.makedirs(config.out_path)

    example = torch.ones(config.batch_size, config.input_c, config.input_h, config.input_w)
    model = getattr(models, config.model)(depth=18, num_classes=config.n_classes)
    model.load(config.torch_model)
    model.eval()
    # output = model(example)
    # print(output)
    traced_script_module = torch.jit.trace(model, example)
    traced_script_module.save(config.script_model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch-size',     type=int,       default=1)
    parser.add_argument('--input-c',        type=int,       default=3)
    parser.add_argument('--input-h',        type=int,       default=224)
    parser.add_argument('--input-w',        type=int,       default=224)
    parser.add_argument('--out-path',       type=str,       default='./output')
    parser.add_argument('--model',          type=str,       default='ResNet')
    parser.add_argument('--n-classes',      type=int,       default=1000)
    parser.add_argument('--torch-model',    type=str,       default='./output/resnet18-5c106cde.pth')
    parser.add_argument('--script-model',   type=str,       default='./output/resnet18.pt')

    config = parser.parse_args()
    args = vars(config)
    print('------------ Options -------------')
    for key, value in sorted(args.items()):
        print('%16.16s: %3.48s' % (str(key), str(value)))
    print('-------------- End ----------------')

    main(config)
    print('End!!')