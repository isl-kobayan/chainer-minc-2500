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

image_size = 362

def makeMeanImage(mean_value):
    mean_image = np.ndarray((3, image_size, image_size), dtype=np.float32)
    for i in range(3): mean_image[i] = mean_value[i]
    return mean_image

def main(args):
    # chainer.set_debug(True)
    # Initialize the model to train
    model = models.archs[args.arch]()
    nowt = datetime.datetime.today()
    if args.initdir is not None:
        outputdir = os.path.join(args.initdir, args.layer)
    else:
        outputdir = os.path.join(args.out, args.arch, 'extract', args.layer)
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)
    # Load the datasets and mean file
    mean = None
    if args.indices is None:
        if args.initdir is not None:
            args.indices = os.path.join(args.initdir, 'top_' + args.layer + '.txt')
        else:
            args.indices = os.path.join(args.out, args.arch, 'extract', 'top_' + args.layer + '.txt')
    if (not args.nobounds) and args.bounds is None:
        if args.initdir is not None:
            args.bounds = os.path.join(args.initdir, 'maxbounds_' + args.layer + '.txt')
        else:
            args.bounds = os.path.join(args.out, args.arch, 'extract', 'maxbounds_' + args.layer + '.txt')
        bounds = np.loadtxt(args.bounds, delimiter="\t", dtype='i')

    val = ppds.PreprocessedDataset(args.val, args.root, mean, model.insize, False, args.indices)
    val_iter = chainer.iterators.SerialIterator(
            val, args.val_batchsize, repeat=False, shuffle=False)


    rows = val.rows
    cols = val.cols
    idx = 0
    for batch in val_iter:
        indices = np.arange(idx, idx + len(batch))
        idx += len(batch)
        if args.nobounds:
            for (i, ba) in zip(indices, batch):
                patch = ba[0]
                patchimg = Image.fromarray(np.uint8(patch[::-1].transpose(1, 2, 0)))
                patchimg.save(os.path.join(outputdir,
                    "{0:0>4}_{1:0>2}.png".format(i % cols, i // cols)))
        else:
            batch_bounds = [bounds[i] for i in indices]
            for (i, ba, bo) in zip(indices, batch, batch_bounds):
                patch = ba[0][:, int(bo[2]):int(bo[3]), int(bo[0]):int(bo[1])]
                patchimg = Image.fromarray(np.uint8(patch[::-1].transpose(1, 2, 0)))
                patchimg.save(os.path.join(outputdir,
                    "{0:0>4}_{1:0>2}.png".format(i % cols, i // cols)))
        #print(batch)

    '''# Set up an optimizer
    optimizer = chainer.optimizers.MomentumSGD()
    optimizer.setup(model)

    # Set up a trainer
    updater = training.StandardUpdater(val_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (1, 'epoch'), outputdir)

    #val_interval = (10 if args.test else int(len(train) / args.batchsize)), 'iteration'
    val_interval = (1, 'iteration')
    #snapshot_interval = (10, 'iteration') if args.test else (2, 'epoch')
    log_interval = (10, 'iteration')

    # Copy the chain with shared parameters to flip 'train' flag only in test
    eval_model = model.copy()
    eval_model.train = False
    val_acquirer = acquirer.Acquirer(val_iter, eval_model, device=args.gpu)
    val_acquirer.layer_rank = eval_model.layer_rank[args.layer]
    val_acquirer.layer_name = args.layer
    val_acquirer.operation = args.operation
    val_acquirer.top = args.top
    val_acquirer.n_features = val.cols
    print(val.cols)
    if 'googlenet' in args.arch:
        val_acquirer.lastname = 'validation/main/loss3'
    trainer.extend(val_acquirer, trigger=val_interval)
    #trainer.extend(extensions.dump_graph('main/loss'))
    #trainer.extend(extensions.snapshot(), trigger=snapshot_interval)
    #trainer.extend(extensions.snapshot_object(
    #    model, 'model_iter_{.updater.iteration}'), trigger=snapshot_interval)
    # Be careful to pass the interval directly to LogReport
    # (it determines when to emit log rather than when to read observations)
    #trainer.extend(extensions.LogReport(trigger=log_interval))
    trainer.extend(extensions.PrintReport([
        'epoch', 'iteration', 'main/loss', 'validation/main/loss',
        'main/accuracy', 'validation/main/accuracy',
    ]), trigger=log_interval)
    trainer.extend(extensions.ProgressBar(update_interval=10))
    #trainer.extend(extensions.ExponentialShift('lr', args.gamma),
    #    trigger=(1, 'epoch'))

    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)

    #if not args.test:
    #    trainer.run()
    #    chainer.serializers.save_npz(outputdir + '/model', model)

    results = val_acquirer(trainer)
    results['outputdir'] = outputdir'''

parser = argparse.ArgumentParser(
    description='Learning convnet from MINC-2500 dataset')
#parser.add_argument('train', help='Path to training image-label list file')
parser.add_argument('val', help='Path to validation image-label list file')
parser.add_argument('--arch', '-a', choices=models.archs.keys(), default='nin',
                    help='Convnet architecture')
parser.add_argument('--layer', '-l', default='conv1',
                    help='layer name')
parser.add_argument('--initdir',
                    help='specify result dir (e.g. result/vgg16/YYYYMMDD-HHmm_bsX)')
parser.add_argument('--indices', '-i',
                    help='indices file name (e.g. top_conv1.txt)')
parser.add_argument('--bounds', '-B',
                    help='bounds file name (e.g. maxbounds_conv1.txt)')
parser.add_argument('--nobounds', default=False, action='store_true',
                    help='get image without cropping (default: False)')
parser.add_argument('--scale', '-s', type=int, default=1,
                    help='filter scale')
parser.add_argument('--pad', '-p', type=int, default=1,
                    help='filter padding')
parser.add_argument('--cols', type=int, default=1,
                    help='columns')
parser.add_argument('--gamma', '-G', type=float, default=1.0,
                    help='gamma')
parser.add_argument('--top', '-t', type=int, default=10,
                    help='gather top n activated images')
parser.add_argument('--loaderjob', '-j', type=int,
                    help='Number of parallel data loading processes')
parser.add_argument('--out', '-o', default='result',
                    help='Output directory')
parser.add_argument('--root', '-R', default='.',
                    help='Root directory path of image files')
parser.add_argument('--val_batchsize', '-b', type=int, default=20,
                    help='Validation minibatch size')
#parser.add_argument('--test', action='store_true')
#parser.set_defaults(test=False)

if __name__ == '__main__':
    args = parser.parse_args()

    val_result = main(args)
    #print('loss\taccuracy')
    #print(str(val_result['validation/main/loss']) + '\t' + str(val_result['validation/main/accuracy']))
