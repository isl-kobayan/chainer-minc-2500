#!/usr/bin/env python
"""dimensionality reduction

Prerequisite: To run this example, execute extrect_features.py.

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
from sklearn.manifold import LocallyLinearEmbedding as LLE
from sklearn.manifold import MDS
from sklearn.manifold import TSNE
from sklearn.manifold import Isomap
from sklearn.decomposition import PCA

def main(args):
    outputdir = os.path.dirname(args.vectors)
    #winidx_path = os.path.join(outputdir,
    #    'cos-distance_' + os.path.basename(args.weights))
    point_path = os.path.splitext(args.vectors)[0] + \
        '_{0}_{1}d-points_it{2}_s{3}.txt'.format(
        args.algorithm, args.components, args.iteration, args.samples)
    fig_path = os.path.splitext(args.vectors)[0] + \
        '_{0}_it{1}_s{2}.eps'.format(args.algorithm, args.iteration, args.samples)

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
    if args.algorithm == 'tsne':
        model = utils.TSNE(n_components=args.components, n_iter=args.iteration,
        n_iter_without_progress=args.preprocessdim, angle=args.angle, metric=args.metric)
    elif args.algorithm == 'mds':
        model = MDS(n_components=args.components, n_jobs=-1)
    elif args.algorithm == 'lle':
        model = LLE(n_components=args.components, n_neighbors = args.neighbors, n_jobs=-1)
    elif args.algorithm == 'isomap':
        model = Isomap(n_components=args.components, n_neighbors = args.neighbors, n_jobs=-1)
    elif args.algorithm == 'pca':
        model = PCA(n_components=args.components)
    #X = model.fit_transform(v[:23*10])
    print('fitting...')
    X = model.fit_transform(np.array(selected_vectors))
    Y = np.asarray([x[1] for x in val])

    if args.algorithm == 'pca':
        pca = PCA(n_components = 100)
        pca.fit(np.array(selected_vectors))
        E = pca.explained_variance_ratio_
        print "explained", E
        print "cumsum E", np.cumsum(E)

    print('drawing...')

    markers=['o', 'x', 'v', '+']

    if args.components == 2:
        plt.figure(2, figsize=(8, 6))
        plt.clf()

        #plt.scatter(X[:, 0], X[:, 1], c=Y[:23*10], cmap=plt.cm.jet)
        #plt.scatter(X[:, 0], X[:, 1], c=np.array(Ys), cmap=plt.cm.jet, label=categories)

        for i in range(C):
            plt.scatter(X[samples_per_c*i:samples_per_c*(i+1), 0],
            X[samples_per_c*i:samples_per_c*(i+1), 1],
            marker=markers[i % len(markers)],
            s = 10,
            color=plt.cm.jet(float(i) / (C-1)), label=categories[i])
        plt.xlabel(args.algorithm + '1')
        plt.ylabel(args.algorithm + '2')
        plt.legend(fontsize=10.25, scatterpoints=1, bbox_to_anchor=(1.05, 1.01), loc='upper left')
        plt.subplots_adjust(right=0.7)
        #plt.show()
        plt.savefig(fig_path)
    elif args.components == 3:
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.set_xlabel("X-axis")
        ax.set_ylabel("Y-axis")
        ax.set_zlabel("Z-axis")
        for i in range(C):
            ax.scatter(
                X[samples_per_c*i:samples_per_c*(i+1), 0],
                X[samples_per_c*i:samples_per_c*(i+1), 1],
                X[samples_per_c*i:samples_per_c*(i+1), 2],
                marker=markers[i % len(markers)],
                s=10,
                c=plt.cm.jet(float(i) / (C-1)), label=categories[i])
        plt.show()

    print(model.get_params())
    # save points
    with open(point_path, 'w') as fp:
        for path, t, p in zip(selected_images, Ys, X):
            fp.write("{0}\t{1}\t{2}\n".format(path, t, '\t'.join(map(str,p))))



parser = argparse.ArgumentParser(
    description='Learning convnet from MINC-2500 dataset')
parser.add_argument('val', help='Path to image-label file')
parser.add_argument('vectors', help='Path to feature file (e.g. fc7.npy)')
parser.add_argument('--algorithm', '-al', default='tsne', choices=('tsne', 'mds', 'lle', 'isomap', 'pca'),
                    help='method of dimensionaly reduction')
parser.add_argument('--categories', '-c', default='categories.txt',
                    help='Path to category list file')
parser.add_argument('--components', '-d', type=int, default=2,
                    help='Path to category list file')
parser.add_argument('--preprocessdim', type=int, default=30,
                    help='Path to category list file')
parser.add_argument('--iteration', '-i', type=int, default=200,
                    help='Learning iteraion')
parser.add_argument('--samples', '-s', type=int, default=200,
                    help='Learning iteraion')
parser.add_argument('--angle', '-a', type=float, default=0.5,
                    help='angle of t-SNE')
parser.add_argument('--perplexity', '-p', type=float, default=30.0,
                    help='perplexity of t-SNE')
parser.add_argument('--epoch', '-E', type=int, default=0,
                    help='Learning epoch')
parser.add_argument('--metric', '-m', default='euclidean',
                    help='distance metric')
parser.add_argument('--neighbors', '-n', type=int, default=5,
                    help='n_neighbors param for Isomap')
parser.add_argument('--initdir',
                    help='specify result dir (e.g. result/vgg16/YYYYMMDD-HHmm_bsX)')
parser.add_argument('--resume',
                    help='specify map data (e.g. fc8_map.npy)')
parser.add_argument('--out', '-o', default='result',
                    help='Output directory')

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
