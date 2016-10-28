#!/usr/bin/env python
from __future__ import print_function
import argparse
import os
import six

models = {
    'alexnet':(
        'http://dl.caffe.berkeleyvision.org/bvlc_alexnet.caffemodel',
        'bvlc_alexnet.caffemodel'),
    'caffenet':(
        'http://dl.caffe.berkeleyvision.org/bvlc_reference_caffenet.caffemodel',
        'bvlc_reference_caffenet.caffemodel'),
    'googlenet':(
        'http://dl.caffe.berkeleyvision.org/bvlc_googlenet.caffemodel',
        'bvlc_googlenet.caffemodel'),
    'googlenetbn':(
        'https://github.com/lim0606/caffe-googlenet-bn/blob/master/snapshots/googlenet_bn_stepsize_6400_iter_1200000.caffemodel?raw=true',
        'googlenet_bn.caffemodel'),
    'nin':(
        'https://www.dropbox.com/s/0cidxafrb2wuwxw/nin_imagenet.caffemodel?dl=1',
        'nin_imagenet.caffemodel'),
    'squeezenet10':(
        'https://github.com/DeepScale/SqueezeNet/blob/master/SqueezeNet_v1.0/squeezenet_v1.0.caffemodel?raw=true',
        'squeezenet_v1.0.caffemodel'),
    'squeezenet11':(
        'https://github.com/DeepScale/SqueezeNet/blob/master/SqueezeNet_v1.1/squeezenet_v1.1.caffemodel?raw=true',
        'squeezenet_v1.1.caffemodel'),
    'vgg16':(
        'http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel',
        'VGG_ILSVRC_16_layers.caffemodel'),
    'vgg19':(
        'http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_19_layers.caffemodel',
        'VGG_ILSVRC_19_layers.caffemodel')
    }


parser = argparse.ArgumentParser(
    description='Download a Caffe reference model')
parser.add_argument('model_type', choices=models.keys(),
                    help='Model type')
args = parser.parse_args()


print('Downloading model file...')
model = models[args.model_type]
six.moves.urllib.request.urlretrieve(model[0], os.path.join('.', 'models', model[1]))
print('Done')
