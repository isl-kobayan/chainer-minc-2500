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
import utils
import utils.finetuning
import preprocessed_dataset as ppds
import evaluator_plus
import datetime
import time
import dataio
import os
import acquirer

image_size = 362

def makeMeanImage(mean_value):
    mean_image = np.ndarray((3, image_size, image_size), dtype=np.float32)
    for i in range(3): mean_image[i] = mean_value[i]
    return mean_image

def main(args):
    chainer.set_debug(True)
    # Initialize the model to train
    model = models.archs[args.arch]()
    if args.finetune and hasattr(model, 'finetuned_model_path'):
        finetuning.load_param(model.finetuned_model_path, model, args.ignore)
    if args.initmodel:
        print('Load model from', args.initmodel)
        chainer.serializers.load_npz(args.initmodel, model)
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()  # Make the GPU current
        #if args.test:
        #cuda.cudnn_enabled = False
        model.to_gpu()

    nowt = datetime.datetime.today()
    outputdir = os.path.join(args.out, args.arch, 'extract')
    if args.initmodel is not None:
        outputdir = os.path.dirname(args.initmodel)
        if args.indices is None:
            args.indices = os.path.join(outputdir, 'top_' + args.layer + '.txt')
    # Load the datasets and mean file
    mean = None
    if hasattr(model, 'mean_value'):
        mean = makeMeanImage(model.mean_value)
    else:
        mean = np.load(args.mean)
    assert mean is not None

    if args.indices is None:
        args.indices = os.path.join(args.out, args.arch, 'extract', 'top_' + args.layer + '.txt')

    #top_path = os.path.join(args.out, args.arch, 'extract', 'top_' + args.layer + '.txt')
    #train = ppds.PreprocessedDataset(args.train, args.root, mean, model.insize)
    val = ppds.PreprocessedDataset(args.val, args.root, mean, model.insize, False, args.indices)
    # These iterators load the images with subprocesses running in parallel to
    # the training/validation.
    #train_iter = chainer.iterators.MultiprocessIterator(
    #    train, args.batchsize, shuffle=False, n_processes=args.loaderjob)
    #val_iter = chainer.iterators.MultiprocessIterator(
    #    val, args.val_batchsize, repeat=False, shuffle=False, n_processes=args.loaderjob)
    val_iter = chainer.iterators.SerialIterator(
            val, args.val_batchsize, repeat=False, shuffle=False)

    # Set up an optimizer
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
    val_acquirer = utils.Acquirer(val_iter, eval_model, device=args.gpu)
    val_acquirer.layer_rank = eval_model.layer_rank[args.layer]
    val_acquirer.layer_name = args.layer
    val_acquirer.operation = args.operation
    val_acquirer.top = args.top
    val_acquirer.n_features = val.cols
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
    results['outputdir'] = outputdir

    #if eval_model.layer_rank[args.layer] == 1:
    #    save_first_conv_filter(os.path.join(outputdir, args.layer),
    #    model[args.layer].W.data, cols = args.cols, pad = args.pad,
    #    scale = args.scale, gamma = args.gamma)

    #if args.test:
    #print(val_acquirer.confmat)
    #categories = dataio.load_categories(args.categories)
    #confmat_csv_name = args.initmodel + '.csv'
    #confmat_fig_name = args.initmodel + '.eps'
    #dataio.save_confmat_csv(confmat_csv_name, val_acquirer.confmat, categories)
    #dataio.save_confmat_fig(confmat_fig_name, val_acquirer.confmat, categories,
    #                        mode="rate", saveFormat="eps")
    return results

parser = argparse.ArgumentParser(
    description='Learning convnet from MINC-2500 dataset')
#parser.add_argument('train', help='Path to training image-label list file')
parser.add_argument('val', help='Path to validation image-label list file')
parser.add_argument('--categories', '-c', default='categories.txt',
                    help='Path to category list file')
parser.add_argument('--arch', '-a', choices=models.archs.keys(), default='nin',
                    help='Convnet architecture')
#parser.add_argument('--batchsize', '-B', type=int, default=32,
#                    help='Learning minibatch size')
#parser.add_argument('--baselr', default=0.001, type=float,
#                    help='Base learning rate')
#parser.add_argument('--gamma', default=0.7, type=float,
#                    help='Base learning rate')
#parser.add_argument('--epoch', '-E', type=int, default=10,
#                    help='Number of epochs to train')
parser.add_argument('--layer', '-l', default='conv1',
                    help='layer name')
parser.add_argument('--indices', '-i',
                    help='indices file name (e.g. top_conv1.txt)')
parser.add_argument('--operation', '-op', choices=('max', 'mean'), default='max',
                    help='operation')
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
parser.add_argument('--gpu', '-g', type=int, default=-1,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--finetune', '-f', default=False, action='store_true',
                    help='do fine-tuning if this flag is set (default: False)')
parser.add_argument('--initmodel',
                    help='Initialize the model from given file')
parser.add_argument('--ignore', nargs='*', default=[],
                    help='Ignored layers in parameter copy')
parser.add_argument('--loaderjob', '-j', type=int,
                    help='Number of parallel data loading processes')
parser.add_argument('--mean', '-m', default='mean.npy',
                    help='Mean file (computed by compute_mean.py)')
parser.add_argument('--resume', '-r', default='',
                    help='Initialize the trainer from given file')
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
    print('loss\taccuracy')
    print(str(val_result['validation/main/loss']) + '\t' + str(val_result['validation/main/accuracy']))
