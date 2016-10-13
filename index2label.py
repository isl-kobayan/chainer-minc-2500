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
from PIL import Image

def main(args):
    if args.initdir is not None:
        outputdir = args.initdir
    else:
        outputdir = os.path.join(args.out, args.arch, 'extract')
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)

    # read val
    with open(args.val) as pairs_file:
        labels = []
        for i, line in enumerate(pairs_file):
            pair = line.strip().split()
            if len(pair) != 2:
                raise ValueError(
                    'invalid format at line {} in file {}'.format(
                        i, args.val))
            labels.append(int(pair[1]))

    if args.indices is None:
        if args.initdir is not None:
            args.indices = os.path.join(args.initdir, 'top_' + args.layer + '.txt')
        else:
            args.indices = os.path.join(args.out, args.arch, 'extract', 'top_' + args.layer + '.txt')

    indices = np.loadtxt(args.indices, delimiter="\t", dtype='i')
    all_labels = labels
    labels = [all_labels[i] for i in indices.flatten()]
    indices2labels = np.asarray(labels).reshape(indices.shape)
    np.savetxt(os.path.join(outputdir, 'labels_' + os.path.basename(args.indices)), indices2labels, delimiter="\t", fmt='%d')

parser = argparse.ArgumentParser(
    description='Learning convnet from MINC-2500 dataset')
parser.add_argument('val', help='Path to validation image-label list file')
parser.add_argument('--indices', '-i',
                    help='indices file name (e.g. top_conv1.txt)')
parser.add_argument('--initdir',
                    help='specify result dir (e.g. result/vgg16/YYYYMMDD-HHmm_bsX)')
parser.add_argument('--out', '-o', default='result',
                    help='Output directory')

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
