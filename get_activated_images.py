#!/usr/bin/env python
"""Example code of learning Flickr Material Dataset (FMD).

Prerequisite: To run this example, crop the center of ILSVRC2012 training and
validation images and scale them to 256x256, and make two lists of space-
separated CSV whose first column is full path to image and second column is
zero-origin label (this format is same as that used by Caffe's ImageDataLayer).

"""
from __future__ import print_function
import argparse
import datetime
import json
import multiprocessing
import os
import random
import sys
import threading
import time

import numpy as np
from PIL import Image
import six
import six.moves.cPickle as pickle
from six.moves import queue

import chainer
from chainer import computational_graph
from chainer import cuda
from chainer import optimizers
from chainer import serializers
import chainer.functions as F

import dataio
import models
import pandas as pd
from PIL import Image

class TrainResult:
    def __init__(self):
        self.valid = False

def get_activates(args):
    if args.gpu >= 0:
        cuda.check_cuda_available()
        print('use GPU')
    else:
        print('use CPU only')

    xp = cuda.cupy if args.gpu >= 0 else np
    # cuda.cudnn_enabled = False
    # Prepare dataset
    val_list = dataio.load_image_list(args.val, args.root)
    val_size = len(val_list)
    assert val_size % args.val_batchsize == 0

    categories = pd.read_csv(args.categories, header=None)

    assert args.layeractivates is not None
    layeractivates = np.load(args.layeractivates)
    assert layeractivates.shape[0] == val_size
    map_size = layeractivates.shape[1]
    #indexes = np.argsort(a)
    df = pd.DataFrame(layeractivates)
    val_path_list = [v[0] for v in val_list]
    val_label_list = [categories.ix[i[1],0] for i in val_list]
    df['path'] = val_path_list
    df['label'] = val_label_list

    outdir = './' + args.arch + '/' + args.layer
    os.makedirs(outdir)

    # Prepare model
    model = models.getModel(args.arch)
    if model is None:
        raise ValueError('Invalid architecture name')
    #if args.finetune and hasattr(model, 'load_param'):
    #    print ('finetune')
    #    model.load_param()

    '''mean_image = None
    if hasattr(model, 'getMean'):
        mean_image = model.getMean()
    else:
        #mean_image = pickle.load(open(args.mean, 'rb'))
        mean_image = np.load(args.mean)

    assert mean_image is not None'''

    print(model.__class__.__name__)
    print('batchsize (validation) : ' + str(args.val_batchsize))
    val_image_pool = [None] * args.top
    impool = multiprocessing.Pool(args.loaderjob)

    for c in six.moves.range(map_size):
        #topacts = df.sort(i, ascending=False)[['path', 'label', i]].iloc[:args.top]
        topacts = df.sort_values(by=[c], ascending=False)[['path', 'label', c]].iloc[:args.top]
        topacts.to_csv(outdir + '/' + "{0:0>4}".format(i) + '.csv', header=None, index=None)
        images = np.zeros((3, model.insize, model.insize * args.top))
        #for n, path in enumerate(topacts['path']):
        #    images[:,:,n*model.insize:(n+1)*model.insize] = \
        #        dataio.read_image(path, model.insize, None, True, False)

        for n, path in enumerate(topacts['path']):
            val_image_pool[n] = impool.apply_async(
                dataio.read_image, (path, model.insize, None, True, False))
        for n, im in enumerate(val_image_pool):
            images[:,:,n*model.insize:(n+1)*model.insize] = im.get()

        pilImg = Image.fromarray(np.uint8(images[::-1].transpose(1, 2, 0)))
        pilImg.save(outdir + '/' + "{0:0>4}".format(c) + '.jpg', 'JPEG', quality=100, optimize=True)
    impool.close()
    impool.join()
    return
    nowt = datetime.datetime.today()
    #outdir = './results/' + args.arch + '_bs' + str(args.batchsize) + '_' + nowt.strftime("%Y%m%d-%H%M")
    #os.makedirs(outdir)
    #args.out = outdir + '/' + args.out + '_' + args.arch
    #args.outstate = outdir + '/' + args.outstate

    if args.gpu >= 0:
        cuda.get_device(args.gpu).use()
        model.to_gpu()

    # Init/Resume
    if args.initmodel:
        print('Load model from', args.initmodel)
        serializers.load_hdf5(args.initmodel, model)
    if args.resume:
        print('Load optimizer state from', args.resume)
        serializers.load_hdf5(args.resume, optimizer)

    # ------------------------------------------------------------------------------
    # This example consists of three threads: data feeder, logger and trainer.
    # These communicate with each other via Queue.
    data_q = queue.Queue(maxsize=1)
    res_q = queue.Queue()
    ret = TrainResult()

    def feed_data():
        # Data feeder
        i = 0
        count = 0

        val_x_batch = np.ndarray(
            (args.val_batchsize, 3, model.insize, model.insize), dtype=np.float32)
        val_y_batch = np.ndarray((args.val_batchsize,), dtype=np.int32)

        val_batch_pool = [None] * args.val_batchsize
        pool = multiprocessing.Pool(args.loaderjob)
        data_q.put('val')
        j = 0
        for path, label in val_list:
            val_batch_pool[j] = pool.apply_async(
                dataio.read_image, (path, model.insize, mean_image, True, False))
            val_y_batch[j] = label
            j += 1

            if j == args.val_batchsize:
                for k, x in enumerate(val_batch_pool):
                    val_x_batch[k] = x.get()
                data_q.put((val_x_batch.copy(), val_y_batch.copy()))
                j = 0
        pool.close()
        pool.join()
        data_q.put('end')

    def log_result():
        # Logger
        testlogfilename=args.out+'/val.log'
        activatescsvfilename=args.out+'/acts_'+args.layer+'.csv'
        activatesnpyfilename=args.out+'/acts_'+args.layer+'.npy'
        train_count = 0
        train_cur_loss = 0
        train_cur_accuracy = 0
        begin_at = time.time()
        val_begin_at = None
        result = None
        Ret = [ret]
        activates=None

        while True:
            result = res_q.get()
            if result == 'end':
                print(file=sys.stderr)
                break
            elif result == 'val':
                print(file=sys.stderr)
                train = False
                val_count = val_loss = val_accuracy = 0
                val_begin_at = time.time()
                continue

            loss, accuracy, max_activates = result
            if activates is None:
                activates = cuda.to_cpu(max_activates)
            else:
                activates = np.r_[activates, cuda.to_cpu(max_activates)]

            val_count += args.val_batchsize
            duration = time.time() - val_begin_at
            throughput = val_count / duration
            sys.stderr.write(
                '\rval   {} batches ({} samples) time: {} ({} images/sec)'
                .format(val_count / args.val_batchsize, val_count,
                        datetime.timedelta(seconds=duration), throughput))

            val_loss += loss
            val_accuracy += accuracy
            #print('accuacy', accuracy)
            if val_count == val_size:
                mean_loss = val_loss * args.val_batchsize / val_size
                mean_error = 1 - val_accuracy * args.val_batchsize / val_size
                print(file=sys.stderr)
                print(json.dumps({'type': 'val', 'iteration': train_count,
                                  'error': mean_error, 'loss': mean_loss}))
                with open(testlogfilename, 'a') as f:
                    f.write(json.dumps({'type': 'val', 'iteration': train_count,
                                        'error': mean_error, 'loss': mean_loss})+'\n')
                Ret[0].val_loss = mean_loss
                Ret[0].val_error = mean_error
                sys.stdout.flush()
        print(activates.shape)
        np.savetxt(activatescsvfilename, activates, delimiter=",")
        np.save(activatesnpyfilename, activates)
        Ret[0].activates = activates

    def train_loop():
        # Trainer
        while True:
            while data_q.empty():
                time.sleep(0.1)
            inp = data_q.get()
            if inp == 'end':  # quit
                res_q.put('end')
                break
            elif inp == 'val':  # start validation
                res_q.put('val')
                model.train = False
                continue

            model.train = False
            volatile = 'off' #if model.train else 'on'
            x = chainer.Variable(xp.asarray(inp[0]), volatile=volatile)
            t = chainer.Variable(xp.asarray(inp[1]), volatile=volatile)

            model(x, t)

            #fc8, = model(inputs={'data': x}, outputs=['fc8'], train=False)
            #model.loss = F.softmax_cross_entropy(fc8, t)
            #model.accuracy = F.accuracy(fc8, t)

            #y, = model(inputs={'data': x}, outputs=['loss3/classifier'], disable=['loss1/ave_pool', 'loss2/ave_pool'], train=False)
            #model.loss = F.softmax_cross_entropy(y, t)
            #model.accuracy = F.accuracy(y, t)

            variable = model.getLayerVariableFromLoss(args.layer)
            #print(model.layer2rank(args.layer))
            #print(variable)
            ax = (2,3) if len(variable.data.shape) == 4 else 1
            max_activates = variable.data.max(axis=ax)
            #max_activates = np.arange(args.val_batchsize * 10).reshape((args.val_batchsize, 10))
            #data = cuda.to_cpu(variable.data)
            #argmax = data.argmax(axis=(1))
            #print(data.shape)
            #print(argmax)

            res_q.put((float(model.loss.data), float(model.accuracy.data), max_activates))
            del x, t,

    # Invoke threads
    feeder = threading.Thread(target=feed_data)
    feeder.daemon = True
    feeder.start()
    logger = threading.Thread(target=log_result)
    logger.daemon = True
    logger.start()

    train_loop()
    feeder.join()
    logger.join()

    # Save final model
    ret.outdir = args.out
    ret.valid = True
    return ret

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Learning convnet from Flickr Material Database')
    parser.add_argument('--val', '-v', default='val.txt',
                        help='Path to validation image-label list file')
    parser.add_argument('--layeractivates', '-n', default=None,
                        help='Path to layer activates file (*.npy)')
    parser.add_argument('--top', '-t', type=int, default=10,
                        help='gets top N images')
    parser.add_argument('--categories', '-c', default='categories.txt',
                        help='Path to the mean file (computed by compute_mean.py)')
    parser.add_argument('--mean', '-m', default='ilsvrc_2012_mean.npy',
                        help='Path to the mean file (computed by compute_mean.py)')
    parser.add_argument('--arch', '-a', default='alex',
                        help='Convnet architecture \
                        (nin, alex, alexbn, vgg16, googlenet, googlenet2, googlenetbn)')
    parser.add_argument('--layer', '-l', default='conv1',
                        help='layer name')
    parser.add_argument('--val_batchsize', '-b', type=int, default=10,
                        help='Validation minibatch size')
    parser.add_argument('--gpu', '-g', default=0, type=int,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--loaderjob', '-j', default=20, type=int,
                        help='Number of parallel data loading processes')
    parser.add_argument('--root', '-r', default='.',
                        help='Root directory path of image files')
    parser.add_argument('--out', '-o', default='model',
                        help='Path to save model on each validation')
    parser.add_argument('--finetune', '-f', default=False, action='store_true',
                        help='do fine-tuning if this flag is set (default: False)')
    parser.add_argument('--initmodel', default='',
                        help='Initialize the model from given file')
    parser.add_argument('--resume', default='',
                        help='Resume the optimization from snapshot')
    args = parser.parse_args()
    get_activates(args)
