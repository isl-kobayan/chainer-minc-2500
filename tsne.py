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
from sompy import SOM
import numpy as np
import matplotlib.pyplot as plt
import dataio
from tqdm import tqdm
from sklearn.manifold import TSNE

def main(args):
    outputdir = os.path.dirname(args.vectors)
    #winidx_path = os.path.join(outputdir,
    #    'cos-distance_' + os.path.basename(args.weights))
    map_path = os.path.splitext(args.vectors)[0] + '_tsne_map.npy'
    winidx_path = os.path.splitext(args.vectors)[0] + '_tsne_winidx.tsv'
    hist_path = os.path.splitext(args.vectors)[0] + '_tsne_hist.tsv'
    mode_path = os.path.splitext(args.vectors)[0] + '_tsne_mode.tsv'
    nearest_path = os.path.splitext(args.vectors)[0] + '_tsne_nearest.tsv'

    print('loading val...')
    val = dataio.load_image_list(args.val)
    categories = dataio.load_categories(args.categories)

    v = np.load(args.vectors)
    N = v.shape[0]
    d = v.shape[1]
    C = len(categories)

    model = TSNE(n_components=2, n_iter=200)
    X = model.fit_transform(v[:23*10])
    Y = np.asarray([x[1] for x in val])
    plt.figure(2, figsize=(8, 6))
    plt.clf()
    plt.scatter(X[:, 0], X[:, 1], c=Y[:23*10], cmap=plt.cm.jet)
    plt.xlabel('tsne1')
    plt.ylabel('tsne2')
    plt.show()




parser = argparse.ArgumentParser(
    description='Learning convnet from MINC-2500 dataset')
parser.add_argument('val', help='Path to image-label file')
parser.add_argument('vectors', help='Path to feature file (e.g. fc7.npy)')
parser.add_argument('--categories', '-c', default='categories.txt',
                    help='Path to category list file')
parser.add_argument('--iteration', '-i', type=int, default=0,
                    help='Learning iteraion')
parser.add_argument('--mapsize', '-m', type=int, nargs=2, default=[10, 10],
                    help='Learning iteraion')
parser.add_argument('--epoch', '-E', type=int, default=0,
                    help='Learning epoch')
parser.add_argument('--initdir',
                    help='specify result dir (e.g. result/vgg16/YYYYMMDD-HHmm_bsX)')
parser.add_argument('--resume',
                    help='specify map data (e.g. fc8_map.npy)')
parser.add_argument('--out', '-o', default='result',
                    help='Output directory')

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
