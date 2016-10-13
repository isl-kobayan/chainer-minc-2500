#!/usr/bin/env python
"""Train convnet for MINC-2500 dataset.

Prerequisite: To run this example, crop the center of ILSVRC2012 training and
validation images and scale them to 256x256, and make two lists of space-
separated CSV whose first column is full path to image and second column is
zero-origin label (this format is same as that used by Caffe's ImageDataLayer).

"""
from __future__ import print_function
import argparse
import random

import numpy as np

import chainer
from chainer import training
from chainer.training import extensions
from chainer import cuda
import models
import preprocessed_dataset as ppds
import datetime
import time
import dataio
import os
from PIL import Image

def main(args):
    # chainer.set_debug(True)
    # Initialize the model to train
    model = models.archs[args.arch]()
    nowt = datetime.datetime.today()
    if args.initdir is not None:
        outputdir = args.initdir
        args.pred = os.path.join(args.initdir, 'pred.txt')
    else:
        outputdir = os.path.join(args.out, args.arch, 'extract')
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)
    categories = dataio.load_categories(args.categories)
    # read val
    with open(args.val) as pairs_file:
        pairs = []
        for i, line in enumerate(pairs_file):
            pair = line.strip().split()
            if len(pair) != 2:
                raise ValueError(
                    'invalid format at line {} in file {}'.format(
                        i, args.val))
            pairs.append((pair[0], int(pair[1])))
    # read pred
    with open(args.pred) as pairs_file:
        preds = []
        for i, line in enumerate(pairs_file):
            pair = line.strip().split()
            if len(pair) != 1:
                raise ValueError(
                    'invalid format at line {} in file {}'.format(
                        i, args.pred))
            preds.append(int(pair[0]))

    results = zip(pairs, preds)
    incorrect_results = [(x[0][0], x[0][1], x[1]) for x in results if x[0][1] != x[1]]
    print(len(incorrect_results))

    with open(os.path.join(outputdir, 'incorrect.txt'), 'w') as f:
        for x in incorrect_results:
            f.write(x[0] + '\t' + categories[x[1]] + '\t' + categories[x[2]] + '\n')

    # gather incorrect images
    img_outdir = os.path.join(outputdir, 'incorrect')
    if not os.path.exists(img_outdir):
        os.makedirs(img_outdir)
    for c in categories:
        if not os.path.exists(os.path.join(img_outdir, c)):
            os.makedirs(os.path.join(img_outdir, c))

    for x in incorrect_results:
        image = Image.open(os.path.join(args.root, x[0]))
        left = (image.size[0] - model.insize) // 2
        top = (image.size[1] - model.insize) // 2
        # get center patch
        crop_img = image.crop((left, top, left + model.insize, top + model.insize))
        crop_img.save(os.path.join(img_outdir, categories[x[1]],
            categories[x[2]] +'_' +  os.path.basename(x[0])))

parser = argparse.ArgumentParser(
    description='Learning convnet from MINC-2500 dataset')
parser.add_argument('val', help='Path to validation image-label list file')
parser.add_argument('--pred', help='Path to validation image-label list file')
parser.add_argument('--arch', '-a', choices=models.archs.keys(), default='nin',
                    help='Convnet architecture')
parser.add_argument('--categories', '-c', default='categories.txt',
                    help='Path to category list file')
parser.add_argument('--initdir',
                    help='specify result dir (e.g. result/vgg16/YYYYMMDD-HHmm_bsX)')
parser.add_argument('--loaderjob', '-j', type=int,
                    help='Number of parallel data loading processes')
parser.add_argument('--root', '-R', default='.',
                    help='Root directory path of image files')
parser.add_argument('--out', '-o', default='result',
                    help='Output directory')

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
