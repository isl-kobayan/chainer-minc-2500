#!/usr/bin/env python
"""Train convnet for MINC-2500 dataset.

Prerequisite: To run this example, crop the center of ILSVRC2012 training and
validation images and scale them to 256x256, and make two lists of space-
separated CSV whose first column is full path to image and second column is
zero-origin label (this format is same as that used by Caffe's ImageDataLayer).

"""
import argparse
import numpy as np
import os
import itertools
import scipy.spatial.distance as dis

def main(args):
    outputdir = os.path.dirname(args.weights)
    #outputpath = os.path.join(outputdir,
    #    'cos-distance_' + os.path.basename(args.weights))
    outputpath = os.path.splitext(args.weights)[0] + '.csv'

    w = np.load(args.weights)
    C = w.shape[0]
    N = w.shape[1]

    d = np.zeros(C**2)

    for i, v in enumerate(itertools.product(w, repeat=2)):
        w1, w2 = v
        d[i] = dis.cosine(w1, w2)

    d = 1.0 - d.reshape((C, C))
    print(d)
    np.savetxt(outputpath, d, delimiter=",")

parser = argparse.ArgumentParser(
    description='Learning convnet from MINC-2500 dataset')
parser.add_argument('weights', help='Path to weight file (W.npy)')
parser.add_argument('--initdir',
                    help='specify result dir (e.g. result/vgg16/YYYYMMDD-HHmm_bsX)')
parser.add_argument('--out', '-o', default='result',
                    help='Output directory')

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
