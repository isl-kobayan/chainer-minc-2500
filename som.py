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

def main(args):
    outputdir = os.path.dirname(args.vectors)
    #winidx_path = os.path.join(outputdir,
    #    'cos-distance_' + os.path.basename(args.weights))
    map_path = os.path.splitext(args.vectors)[0] + '_som_map.npy'
    winidx_path = os.path.splitext(args.vectors)[0] + '_som_winidx.tsv'
    hist_path = os.path.splitext(args.vectors)[0] + '_som_hist.tsv'
    mode_path = os.path.splitext(args.vectors)[0] + '_som_mode.tsv'
    nearest_path = os.path.splitext(args.vectors)[0] + '_som_nearest.tsv'

    print('loading val...')
    val = dataio.load_image_list(args.val)
    categories = dataio.load_categories(args.categories)

    v = np.load(args.vectors)
    N = v.shape[0]
    d = v.shape[1]
    C = len(categories)

    def train(som, n):
        #pbar = tqdm(total=n)
        for i in tqdm(range(n)):
            r = np.random.randint(0, som.input_num)
            data = som.input_layer[r]
            win_idx = som._get_winner_node(data)
            som._update(win_idx, data, i)
            #pbar.update(1)
        #pbar.close()
        return som.output_layer.reshape((som.shape[1], som.shape[0], som.input_dim))

    output_shape = None
    if not args.resume:
        print('training...')
        output_shape = (args.mapsize[0], args.mapsize[1])
        som = SOM(output_shape, v)
        som.set_parameter(neighbor=0.1, learning_rate=0.2)

        iteration = N
        if args.epoch != 0:
            iteration = N * args.epoch
        elif args.iteration != 0:
            iteration = args.iteration

            #output_map = som.train(1)
        output_map = train(som, iteration)
        np.save(map_path, output_map)
    else:
        output_map = np.load(args.resume)
        print(output_map.shape)
        output_shape = (output_map.shape[0], output_map.shape[1])
        som = SOM(output_shape, v)
        som.set_parameter(neighbor=0.1, learning_rate=0.2)
        som.output_layer = output_map.reshape(
            (output_shape[0] * output_shape[1], -1))

    print('testing...')
    hist_output_map = np.zeros((output_shape[0], output_shape[1], C),
                            dtype=np.int32)
    pbar = tqdm(total=N)
    with open(winidx_path, 'w') as f:
        for i, (pair, vv) in enumerate(zip(val, v)):
            idx = som._get_winner_node(vv)
            hist_output_map[idx[0], idx[1], pair[1]] += 1
            #print(idx)
            f.write(str(idx[0]) + '\t' + str(idx[1]) + '\n')
            pbar.update(1)
    pbar.close()
    np.savetxt(hist_path, hist_output_map.reshape((output_shape[0], -1)),
                delimiter='\t', fmt='%d')
    mode_category = hist_output_map.argmax(axis=2)
    print(mode_category)
    np.savetxt(mode_path, mode_category, delimiter='\t', fmt='%d')

    def get_nearest(som, input_data):
        nearest_idx = np.zeros(som.output_layer.shape[0])
        for i, data in enumerate(tqdm(som.output_layer)):
            sub = input_data - data
            dis = np.linalg.norm(sub, axis=1)
            nearest_idx[i] = np.argmin(dis)
        return nearest_idx.reshape(som.shape[1], som.shape[0])

    nearest_idx = get_nearest(som, v)
    np.savetxt(nearest_path, nearest_idx, delimiter='\t', fmt='%d')

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
