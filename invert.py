#!/usr/bin/env python
"""Train convnet for MINC-2500 dataset.

Prerequisite: To run this example, put MINC-2500 dataset ("minc-2500" direcotry)
              into this direcotry.
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
import preprocessed_dataset as ppds
import datetime
import time
import os
import chainer.functions as F
from tqdm import tqdm

image_size = 362

optimizers = {'momentumsgd':chainer.optimizers.MomentumSGD,
                'adagrad':chainer.optimizers.AdaGrad,
                'adadelta':chainer.optimizers.AdaDelta,
                'adam':chainer.optimizers.Adam}

def makeMeanImage(mean_value):
    mean_image = np.ndarray((3, image_size, image_size), dtype=np.float32)
    for i in range(3): mean_image[i] = mean_value[i]
    return mean_image

class DummyDataset(chainer.dataset.DatasetMixin):
    def __init__(self, size=1):
        self.size = size
    def __len__(self):
        return self.size
    def get_example(self, i):
        return (i)

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
    #outputdir = args.out + '/' + args.arch + '/' + nowt.strftime("%Y%m%d-%H%M")  + '_bs' +  str(args.batchsize)
    outputdir = args.out
    if args.test and args.initmodel is not None:
        outputdir = os.path.dirname(args.initmodel)
    # Load the datasets and mean file
    mean = None
    if hasattr(model, 'mean_value'):
        mean = makeMeanImage(model.mean_value)
    else:
        mean = np.load(args.mean)
    assert mean is not None

    model.train = False
    insize = model.insize
    x_data = np.random.randn(3, insize, insize).astype(np.float32)[np.newaxis]
    #x_data = x_data * 10
    x0_sigma = 27098.11571533
    #x_data = x_data / np.linalg.norm(x_data) * x0_sigma
    x_data = x_data * 20
    print(x_data.max(), x_data.min())
    if args.infile:
        x_data = utils.io.read_image(args.infile, insize, mean)[np.newaxis]

    #print(x_data)
    inverter = utils.Inverter(model, args.label, x_data,
        beta=args.beta, p=args.norm_p,
        lambda_a=args.lambda_a, lambda_tv=args.lambda_tv, lambda_lp=args.lambda_lp)
    # Set up an optimizer
    #optimizer = optimizers[args.opt]()
    #optimizer = utils.LBFGS()
    optimizer = chainer.optimizers.RMSprop()
    #if args.opt == 'momentumsgd':
    if hasattr(optimizer, 'lr'):
        optimizer.lr = args.baselr
    if hasattr(optimizer, 'momentum'):
        optimizer.momentum = args.momentum

    if args.gpu >= 0:
        inverter.to_gpu()

    optimizer.setup(inverter)

    #print (model.fc8.W.data)

    # Set up a trainer
    dummy_train_dataset = DummyDataset(args.iteration)
    dummy_val_dataset = DummyDataset()
    dummy_train_iter = chainer.iterators.SerialIterator(dummy_train_dataset, 1)
    dummy_val_iter = chainer.iterators.SerialIterator(dummy_val_dataset, 1, repeat=False)
    updater = training.StandardUpdater(dummy_train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (args.iteration, 'iteration'), outputdir)
    trainer.reporter.add_observer('model', inverter.model)
    '''trainer.reporter.add_observers(
                'model', model.namedlinks(skipself=True))'''

    #val_interval = (10 if args.test else int(len(train) / args.batchsize)), 'iteration'
    val_interval = 100, 'iteration'
    snapshot_interval = 100, 'iteration'
    log_interval = 100, 'iteration'

    # Copy the chain with shared parameters to flip 'train' flag only in test
    eval_inverter = inverter.copy()
    eval_inverter.train = False
    val_evaluator = extensions.Evaluator(dummy_val_iter, eval_inverter, device=args.gpu)
    #trainer.extend(val_evaluator, trigger=val_interval)
    trainer.extend(extensions.dump_graph('main/inv_loss'))
    #trainer.extend(extensions.snapshot(), trigger=snapshot_interval)
    #trainer.extend(extensions.snapshot_object(
    #    inverter, 'model_iter_{.updater.iteration}'), trigger=snapshot_interval)
    trainer.extend(utils.ImageSnapshot(
        inverter, 'img', mean,
        'img' + str(args.label) +  '_iter_{.updater.iteration}.png'), trigger=snapshot_interval)
    # Be careful to pass the interval directly to LogReport
    # (it determines when to emit log rather than when to read observations)
    trainer.extend(extensions.LogReport(trigger=log_interval))
    trainer.extend(extensions.PrintReport([
        'epoch', 'iteration', 'main/inv_loss', 'validation/main/inv_loss',
        'main/activation', 'validation/main/activation',
        'main/tv', 'validation/main/tv',
        'main/lp', 'validation/main/lp',
    ]), trigger=log_interval)
    trainer.extend(extensions.ProgressBar(update_interval=10))
    if args.opt == 'momentumsgd':
        trainer.extend(extensions.ExponentialShift('lr', args.gamma),
            trigger=(1, 'iteration'))

    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)

    if not args.test:
        trainer.run()
        #chainer.serializers.save_npz(outputdir + '/model', inverter)
        #with open(outputdir + '/args.txt', 'w') as o:
            #print(args, file=o)

    #results = val_evaluator(trainer)
    #results['outputdir'] = outputdir
    result = utils.io.deprocess(cuda.to_cpu(inverter.img.W.data)[0], mean)
    result.save('result.png')
    #print(inverter.img.W.data)
    #print(inverter.Wh_data)
    #print (model.fc8.W.data)
    print(inverter.img.W.data.max(), inverter.img.W.data.min())
    return result

parser = argparse.ArgumentParser(
    description='Learning convnet from MINC-2500 dataset')
parser.add_argument('label', type=int, default=0, help='label number of inversion')
parser.add_argument('--infile', help='initalize invert map by image')
parser.add_argument('--categories', '-c', default='categories.txt',
                    help='Path to category list file')
parser.add_argument('--arch', '-a', choices=models.archs.keys(), default='nin',
                    help='Convnet architecture')
parser.add_argument('--batchsize', '-B', type=int, default=32,
                    help='Learning minibatch size')
parser.add_argument('--baselr', default=1, type=float,
                    help='Base learning rate')
parser.add_argument('--gamma', default=0.999, type=float,
                    help='Base learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='momentum')
parser.add_argument('--iteration', '-i', type=int, default=1000,
                    help='Number of epochs to train')
parser.add_argument('--beta', default=2, type=float,
                    help='beta value of tv_norm')
parser.add_argument('--norm_p', default=6, type=float,
                    help='pth norm of Lp norm L^p')
parser.add_argument('--lambda_a', default=1, type=float,
                    help='lambda_a')
parser.add_argument('--lambda_tv', default=10, type=float,
                    help='lambda_tv')
parser.add_argument('--lambda_lp', default=10, type=float,
                    help='lambda_lp')
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
    invert_image = main(args)
