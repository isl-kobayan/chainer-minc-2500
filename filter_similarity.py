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
import evaluator_plus
import datetime
import time
import dataio
import os
import itertools
import scipy.spatial.distance as dis
import six
import utils.finetuning
from tqdm import tqdm

def save_list(path, lst):
    with open(path, 'w') as f:
        for p in lst:
            f.write("{0}\t{1}\t{2}\n".format(p[0], p[1], p[2]))

def euclidean_distance_matrix(data):
    N = data.shape[0]
    d = np.zeros(N**2).reshape((N, N))
    for a, b in tqdm(itertools.combinations(six.moves.range(N), 2)):
        d[a, b] = dis.euclidean(data[a], data[b])
        d[b, a] = d[a, b]
    return d

def cos_distance_matrix(data):
    N = data.shape[0]
    d = np.zeros(N**2).reshape((N, N))
    norm = np.linalg.norm(data, axis=1)
    for a, b in tqdm(itertools.combinations(six.moves.range(N), 2)):
        d[a, b] = 1 - np.dot(data[a], data[b]) / (norm[a] * norm[b])
        d[b, a] = d[a, b]
    return d

def main(args):
    chainer.set_debug(True)
    # Initialize the model to train
    model = models.archs[args.arch]()
    if args.finetune and hasattr(model, 'finetuned_model_path'):
        utils.finetuning.load_param(model.finetuned_model_path, model)
    if args.initmodel:
        print('Load model from', args.initmodel)
        chainer.serializers.load_npz(args.initmodel, model)

    outputdir = os.path.join('.', args.out, args.arch)
    if args.initmodel:
        outputdir = os.path.dirname(args.initmodel)
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)

    for layer in args.layer:
        outputpath = os.path.join(outputdir,
            "{0}_distance_{1}.npy".format(layer, args.dis))
        similar_path = os.path.join(outputdir,
            "{0}_similar_{1}.tsv".format(layer, args.dis))
        best_path = os.path.join(outputdir,
            "{0}_best{1}_{2}.tsv".format(layer, args.best, args.dis))

        W, b = model[layer].W.data, model[layer].b.data
        if args.withbias:
            params = np.hstack((W.reshape((W.shape[0], -1)), b.reshape((b.shape[0], -1))))
        else:
            params = W.reshape((W.shape[0], -1))

        print("{0}\tW: {1}\tb: {2}".format(layer, W.shape, b.shape))

        N = params.shape[0]
        #d = np.zeros(N**2).reshape((N, N))
        '''for i, p in enumerate(tqdm(itertools.product(params, repeat=2))):
            w1, w2 = p
            if args.dis == 'cos':
                d[i] = dis.cosine(w1, w2)
            elif args.dis == 'euclidean':
                d[i] = dis.euclidean(w1, w2)'''
        if args.dis == 'cos':
            d = cos_distance_matrix(params)
        elif args.dis == 'euclidean':
            d = euclidean_distance_matrix(params)

        #d = d.reshape((N, N))
        np.savetxt(outputpath, d, delimiter=",")

        d_list = [(int(x[0]), int(x[1]), d[x[0],x[1]]) for x in
            itertools.combinations(six.moves.range(N), 2)]

        if args.threshold:
            similar_list = filter((lambda x: x[2] < args.threshold), d_list)
            save_list(similar_path, similar_list)

        if args.best:
            order_by_disance_ascending = sorted(d_list, key=lambda x: x[2])
            save_list(best_path, order_by_disance_ascending[0:args.best])

parser = argparse.ArgumentParser(
    description='calculate filter similarity')
parser.add_argument('--arch', '-a', choices=models.archs.keys(), default='nin',
                    help='Convnet architecture')
parser.add_argument('--gpu', '-g', type=int, default=-1,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--finetune', '-f', default=False, action='store_true',
                    help='do fine-tuning if this flag is set (default: False)')
parser.add_argument('--initmodel',
                    help='Initialize the model from given file')
parser.add_argument('--layer', '-l', nargs='+',
                    help='layer name')
parser.add_argument('--dis', '-d', choices=('cos', 'euclidean'),default='cos',
                    help='layer name')
parser.add_argument('--threshold', '-t', type=float,
                    help='threshold')
parser.add_argument('--best', '-b', type=int,
                    help='output best ')
parser.add_argument('--out', '-o', default='result',
                    help='Output directory')
parser.add_argument('--withbias', action='store_true')
parser.set_defaults(test=False)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
