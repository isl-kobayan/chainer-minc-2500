#!/usr/bin/env python
from __future__ import print_function
import argparse

import six


parser = argparse.ArgumentParser(
    description='Download a Caffe reference model')
parser.add_argument('model_type', choices=('alexnet', 'caffenet', 'googlenet', 'nin', 'squeezenet10', 'squeezenet11','vgg16', 'vgg19'),
                    help='Model type (alexnet, caffenet, googlenet, nin, squeezenet10, squeezenet11, vgg16, vgg19)')
args = parser.parse_args()

if args.model_type == 'alexnet':
    url = 'http://dl.caffe.berkeleyvision.org/bvlc_alexnet.caffemodel'
    name = 'bvlc_alexnet.caffemodel'
elif args.model_type == 'caffenet':
    url = 'http://dl.caffe.berkeleyvision.org/' \
          'bvlc_reference_caffenet.caffemodel'
    name = 'bvlc_reference_caffenet.caffemodel'
elif args.model_type == 'googlenet':
    url = 'http://dl.caffe.berkeleyvision.org/bvlc_googlenet.caffemodel'
    name = 'bvlc_googlenet.caffemodel'
elif args.model_type == 'nin':
    url = 'https://www.dropbox.com/s/0cidxafrb2wuwxw/nin_imagenet.caffemodel?dl=1'
    name = 'nin_imagenet.caffemodel'
elif args.model_type == 'squeezenet10':
    url = 'https://github.com/DeepScale/SqueezeNet/blob/master/SqueezeNet_v1.0/squeezenet_v1.0.caffemodel?raw=true'
    name = 'squeezenet_v1.0.caffemodel'
elif args.model_type == 'squeezenet11':
    url = 'https://github.com/DeepScale/SqueezeNet/blob/master/SqueezeNet_v1.1/squeezenet_v1.1.caffemodel?raw=true'
    name = 'squeezenet_v1.1.caffemodel'
elif args.model_type == 'vgg16':
    url = 'http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel'
    name = 'VGG_ILSVRC_16_layers.caffemodel'
elif args.model_type == 'vgg19':
    url = 'http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_19_layers.caffemodel'
    name = 'VGG_ILSVRC_19_layers.caffemodel'
else:
    raise RuntimeError('Invalid model type. Choose from '
                       'alexnet, caffenet, googlenet, nin, squeezenet10, squeezenet11, vgg16 and vgg19.')

print('Downloading model file...')
six.moves.urllib.request.urlretrieve(url, './models/' + name)
print('Done')
