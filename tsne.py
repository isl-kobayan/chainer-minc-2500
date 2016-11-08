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
import utils
from tqdm import tqdm
from sklearn.manifold import TSNE

def main(args):
    outputdir = os.path.dirname(args.vectors)
    #winidx_path = os.path.join(outputdir,
    #    'cos-distance_' + os.path.basename(args.weights))
    point_path = os.path.splitext(args.vectors)[0] + \
        '_tsne_points_it{0}_s{1}.txt'.format(args.iteration, args.samples)
    winidx_path = os.path.splitext(args.vectors)[0] + '_tsne_winidx.tsv'
    hist_path = os.path.splitext(args.vectors)[0] + '_tsne_hist.tsv'
    mode_path = os.path.splitext(args.vectors)[0] + '_tsne_mode.tsv'
    fig_path = os.path.splitext(args.vectors)[0] + \
        '_tsne_it{0}_s{1}.eps'.format(args.iteration, args.samples)

    print('loading val...')
    val = utils.io.load_image_list(args.val)
    categories = utils.io.load_categories(args.categories)

    v = np.load(args.vectors)
    N = v.shape[0]
    d = v.shape[1]
    C = len(categories)
    NperC = N//C

    samples_per_c = args.samples
    random_order =  np.random.permutation(NperC)
    selected_vectors = []
    selected_images = []
    Ys = []
    for i in range(C):
        selected_vectors.extend([v[i*NperC + ii] for ii in random_order[:samples_per_c]])
        selected_images.extend([val[i*NperC + ii][0] for ii in random_order[:samples_per_c]])
        Ys.extend([val[i*NperC + ii][1] for ii in random_order[:samples_per_c]])

    #print(selected_vectors)
    #print(Ys)
    model = TSNE(n_components=2, n_iter=args.iteration)
    #X = model.fit_transform(v[:23*10])
    print('fitting...')
    X = model.fit_transform(np.array(selected_vectors))
    Y = np.asarray([x[1] for x in val])
    plt.figure(2, figsize=(8, 6))
    plt.clf()

    markers=['o', 'x', 'v', '+']

    #plt.scatter(X[:, 0], X[:, 1], c=Y[:23*10], cmap=plt.cm.jet)
    #plt.scatter(X[:, 0], X[:, 1], c=np.array(Ys), cmap=plt.cm.jet, label=categories)
    for i in range(C):
        plt.scatter(X[samples_per_c*i:samples_per_c*(i+1), 0],
        X[samples_per_c*i:samples_per_c*(i+1), 1],
        marker=markers[i % len(markers)],
        s = 10,
        color=plt.cm.jet(float(i) / (C-1)), label=categories[i])
    plt.xlabel('tsne1')
    plt.ylabel('tsne2')
    plt.legend(fontsize=10.25, scatterpoints=1, bbox_to_anchor=(1.05, 1.01), loc='upper left')
    plt.subplots_adjust(right=0.7)
    #plt.show()
    plt.savefig(fig_path)
    print(model.get_params())

    # save points
    with open(point_path, 'w') as fp:
        for path, t, p in zip(selected_images, Ys, X):
            fp.write("{0}\t{1}\t{2}\t{3}\n".format(path, t, p[0], p[1]))



parser = argparse.ArgumentParser(
    description='Learning convnet from MINC-2500 dataset')
parser.add_argument('val', help='Path to image-label file')
parser.add_argument('vectors', help='Path to feature file (e.g. fc7.npy)')
parser.add_argument('--categories', '-c', default='categories.txt',
                    help='Path to category list file')
parser.add_argument('--iteration', '-i', type=int, default=200,
                    help='Learning iteraion')
parser.add_argument('--samples', '-s', type=int, default=200,
                    help='Learning iteraion')
parser.add_argument('--angle', '-a', type=float, default=0.5,
                    help='angle of t-SNE')
parser.add_argument('--perplexity', '-p', type=int, default=30,
                    help='perplexity of t-SNE')
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
