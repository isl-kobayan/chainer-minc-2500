#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" visualizing filters of convolutional layer.
最初の畳み込み層のフィルタを可視化します。

Prerequisite: To run this program, prepare model's weights data with hdf5 format.
train_fmd.py等で生成される、hdf5形式で保存された重みデータ（model_googlenet）を用意してください。

"""
from __future__ import print_function
import argparse
import datetime
import random
import sys
import time
import datetime
import locale
import os

import numpy as np
from PIL import Image
import six

import finetuning
import models

def save_first_conv_filter(outdir, W, cols=1, pad=1, scale=1, gamma=1.0):
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    out_ch, in_ch, height, width = W.shape
    Wmin, Wmax = np.min(W), np.max(W)
    Wrange = Wmax - Wmin
    print('min, max = ', Wmin, Wmax)

    rows = (int)((out_ch + cols - 1) / cols)

    w_step = width * scale + pad
    h_step = height * scale + pad

    # all filters
    all_img_width = w_step * cols + pad
    all_img_height = h_step * rows + pad
    all_img = Image.new('RGB', (all_img_width, all_img_height), (255, 255, 255))

    # if number of input channels is 3, visualize filter with RGB
    if in_ch == 3:
        for i in six.moves.range(0, out_ch):
            filter_data = (((W[i][::-1].transpose(1, 2, 0) - Wmin) / Wrange) ** gamma) * 255
            img = Image.fromarray(np.uint8(filter_data))
            if args.scale > 1:
                img = img.resize((width * scale, height * scale), Image.NEAREST)
            all_img.paste(img, (pad + (i % cols) * w_step, pad + (int)(i // cols) * h_step))
            img.save(os.path.join(outdir, 'w' + str(i) + '.png'))
        all_img.save(os.path.join(outdir, 'filters.png'))
    else:
        for i in six.moves.range(0, out_ch):
            for j in six.moves.range(0, in_ch):
                filter_data = (((W[i][j] - Wmin) / Wrange) ** gamma) * 255
                img = Image.fromarray(np.uint8(filter_data))
                img.save(os.path.join(outdir, 'w' + str(i) + '_' + str(j) + '.png'))

parser = argparse.ArgumentParser(
                    description='Learning Flickr Material Database')
parser.add_argument('--arch', '-a', choices=models.archs.keys(), default='nin',
                    help='Convnet architecture')
parser.add_argument('--initmodel',
                    help='Initialize the model from given file')
parser.add_argument('--layer', '-l', default='conv1',
                    help='layer name (example: conv1, fc6, etc)')
parser.add_argument('--scale', '-s', type=int, default=1,
                    help='filter scale')
parser.add_argument('--pad', '-p', type=int, default=1,
                    help='filter padding')
parser.add_argument('--cols', '-c', type=int, default=1,
                    help='columns')
parser.add_argument('--gamma', '-G', type=float, default=1.0,
                    help='gamma')
parser.add_argument('--finetune', '-f', default=False, action='store_true',
                    help='Visualize filters before fine-tuning if True (default: False)')
parser.add_argument('--out', '-o', default='result',
                    help='Output directory')

if __name__ == '__main__':
    args = parser.parse_args()
    # Prepare model
    model = models.archs[args.arch]()
    if args.finetune and hasattr(model, 'finetuned_model_path'):
        finetuning.load_param(model.finetuned_model_path, model)
    if args.initmodel:
        print('Load model from', args.initmodel)
        chainer.serializers.load_npz(args.initmodel, model)

    layer_without_special_char = args.layer.replace('/', '_')
    outputdir = os.path.join(args.out, args.arch, layer_without_special_char)
    if args.initmodel is not None:
        outputdir = os.path.join(os.path.dirname(args.initmodel),
                                    layer_without_special_char)

    save_first_conv_filter(outputdir, model[args.layer].W.data,
                            cols = args.cols, pad = args.pad,
                            scale = args.scale, gamma = args.gamma)
