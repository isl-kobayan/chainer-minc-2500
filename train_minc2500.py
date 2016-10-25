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
import utils.evaluator_plus
import datetime
import time
import dataio
import os

image_size = 362

optimizers = {'momentumsgd':chainer.optimizers.MomentumSGD,
                'adagrad':chainer.optimizers.AdaGrad,
                'adadelta':chainer.optimizers.AdaDelta,
                'adam':chainer.optimizers.Adam}

def makeMeanImage(mean_value):
    mean_image = np.ndarray((3, image_size, image_size), dtype=np.float32)
    for i in range(3): mean_image[i] = mean_value[i]
    return mean_image

def main(args):
    # Initialize the model to train
    model = models.archs[args.arch]()
    if args.finetune and hasattr(model, 'finetuned_model_path'):
        utils.finetuning.load_param(model.finetuned_model_path, model, args.ignore)
        #model.finetune = True

    if args.initmodel:
        print('Load model from', args.initmodel)
        chainer.serializers.load_npz(args.initmodel, model)
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        model.to_gpu()

    nowt = datetime.datetime.today()
    outputdir = args.out + '/' + args.arch + '/' + nowt.strftime("%Y%m%d-%H%M")  + '_bs' +  str(args.batchsize)
    if args.test and args.initmodel is not None:
        outputdir = os.path.dirname(args.initmodel)
    # Load the datasets and mean file
    mean = None
    if hasattr(model, 'mean_value'):
        mean = makeMeanImage(model.mean_value)
    else:
        mean = np.load(args.mean)
    assert mean is not None

    train = ppds.PreprocessedDataset(args.train, args.root, mean, model.insize)
    val = ppds.PreprocessedDataset(args.val, args.root, mean, model.insize, False)
    # These iterators load the images with subprocesses running in parallel to
    # the training/validation.
    train_iter = chainer.iterators.MultiprocessIterator(
        train, args.batchsize, shuffle=False, n_processes=args.loaderjob)
    #val_iter = chainer.iterators.MultiprocessIterator(
    #    val, args.val_batchsize, repeat=False, shuffle=False, n_processes=args.loaderjob)
    val_iter = chainer.iterators.SerialIterator(
            val, args.val_batchsize, repeat=False, shuffle=False)

    # Set up an optimizer
    optimizer = optimizers[args.opt]()
    #if args.opt == 'momentumsgd':
    if hasattr(optimizer, 'lr'):
        optimizer.lr = args.baselr
    if hasattr(optimizer, 'momentum'):
        optimizer.momentum = args.momentum
    optimizer.setup(model)

    # Set up a trainer
    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), outputdir)

    #val_interval = (10 if args.test else int(len(train) / args.batchsize)), 'iteration'
    val_interval = (10, 'iteration') if args.test else (1, 'epoch')
    snapshot_interval = (10, 'iteration') if args.test else (4, 'epoch')
    log_interval = (10 if args.test else 200), 'iteration'

    # Copy the chain with shared parameters to flip 'train' flag only in test
    eval_model = model.copy()
    eval_model.train = False
    if not args.test:
        val_evaluator = extensions.Evaluator(val_iter, eval_model, device=args.gpu)
    else:
        val_evaluator = utils.EvaluatorPlus(val_iter, eval_model, device=args.gpu)
        if 'googlenet' in args.arch:
            val_evaluator.lastname = 'validation/main/loss3'
    trainer.extend(val_evaluator, trigger=val_interval)
    trainer.extend(extensions.dump_graph('main/loss'))
    trainer.extend(extensions.snapshot(), trigger=snapshot_interval)
    trainer.extend(extensions.snapshot_object(
        model, 'model_iter_{.updater.iteration}'), trigger=snapshot_interval)
    # Be careful to pass the interval directly to LogReport
    # (it determines when to emit log rather than when to read observations)
    trainer.extend(extensions.LogReport(trigger=log_interval))
    trainer.extend(extensions.PrintReport([
        'epoch', 'iteration', 'main/loss', 'validation/main/loss',
        'main/accuracy', 'validation/main/accuracy',
    ]), trigger=log_interval)
    trainer.extend(extensions.ProgressBar(update_interval=10))
    if args.opt == 'momentumsgd':
        trainer.extend(extensions.ExponentialShift('lr', args.gamma),
            trigger=(1, 'epoch'))

    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)

    if not args.test:
        trainer.run()
        chainer.serializers.save_npz(outputdir + '/model', model)
        with open(outputdir + '/args.txt', 'w') as o:
            print(args, file=o)

    results = val_evaluator(trainer)
    results['outputdir'] = outputdir

    if args.test:
        print(val_evaluator.confmat)
        categories = dataio.load_categories(args.categories)
        confmat_csv_name = args.initmodel + '.csv'
        confmat_fig_name = args.initmodel + '.eps'
        dataio.save_confmat_csv(confmat_csv_name, val_evaluator.confmat, categories)
        dataio.save_confmat_fig(confmat_fig_name, val_evaluator.confmat, categories,
                                mode="rate", saveFormat="eps")
    return results

parser = argparse.ArgumentParser(
    description='Learning convnet from MINC-2500 dataset')
parser.add_argument('train', help='Path to training image-label list file')
parser.add_argument('val', help='Path to validation image-label list file')
parser.add_argument('--categories', '-c', default='categories.txt',
                    help='Path to category list file')
parser.add_argument('--arch', '-a', choices=models.archs.keys(), default='nin',
                    help='Convnet architecture')
parser.add_argument('--batchsize', '-B', type=int, default=32,
                    help='Learning minibatch size')
parser.add_argument('--baselr', default=0.001, type=float,
                    help='Base learning rate')
parser.add_argument('--gamma', default=0.7, type=float,
                    help='Base learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='momentum')
parser.add_argument('--epoch', '-E', type=int, default=10,
                    help='Number of epochs to train')
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
parser.add_argument('--opt', choices=optimizers.keys(), default='momentumsgd',
                    help='optimizer')
parser.add_argument('--out', '-o', default='result',
                    help='Output directory')
parser.add_argument('--root', '-R', default='.',
                    help='Root directory path of image files')
parser.add_argument('--val_batchsize', '-b', type=int, default=20,
                    help='Validation minibatch size')
parser.add_argument('--test', action='store_true')
parser.set_defaults(test=False)

if __name__ == '__main__':
    args = parser.parse_args()
    val_result = main(args)
    print("loss\taccuracy\n{0}\t{1}".format(
        val_result['validation/main/loss'],
        val_result['validation/main/accuracy']))
