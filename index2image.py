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
from PIL import ImageDraw
import models
from matplotlib import cm
import utils

def main(args):
    model = models.archs[args.arch]()
    insize = model.insize
    tiled_image_path = os.path.splitext(args.indices)[0] + '.jpg'
    label_path = os.path.splitext(args.indices)[0] + '_label.tsv'

    val = utils.io.load_image_list(args.val)
    indices = np.loadtxt(args.indices, delimiter="\t", dtype='i')
    rows = indices.shape[0]
    cols = indices.shape[1]
    paths = [val[i][0] for i in indices.flatten()]
    labels = [val[i][1] for i in indices.flatten()]
    label_map = np.asarray(labels, dtype=np.int32).reshape(indices.shape)
    categories = utils.io.load_categories(args.categories)
    C = len(categories)
    print(label_map)

    tiled_image = Image.new("RGB", (cols * insize, rows * insize))
    draw = ImageDraw.Draw(tiled_image)
    lw = 10
    for i, (path, label) in enumerate(zip(paths, labels)):
        r, c = i // cols, int(i % cols)
        x, y = c * insize, r * insize
        img = Image.open(os.path.join(args.root, path)).convert('RGB')
        tiled_image.paste(img, (x, y))
        color = tuple([int(cl * 255) for cl in cm.jet(label / float(C-1.0))])
        # draw rectangle with specified line width
        #draw.rectangle(((x, y),(x+insize, y+insize)), outline=color)
        if r == 0 or label_map[r-1, c] != label:
            draw.rectangle(((x, y),(x+insize, y+lw)), outline=color, fill=color)
        if c == 0 or label_map[r, c-1] != label:
            draw.rectangle(((x, y),(x+lw, y+insize)), outline=color, fill=color)
        if r == rows-1 or label_map[r+1, c] != label:
            draw.rectangle(((x, y+insize-lw),(x+insize, y+insize)), outline=color, fill=color)
        if c == cols-1 or label_map[r, c+1] != label:
            draw.rectangle(((x+insize-lw, y),(x+insize, y+insize)), outline=color, fill=color)
    np.savetxt(label_path, label_map, delimiter="\t", fmt="%d")
    tiled_image.save(tiled_image_path, 'JPEG', quality=100, optimize=True)

parser = argparse.ArgumentParser(
    description='Learning convnet from MINC-2500 dataset')
parser.add_argument('val', help='Path to validation image-label list file')
parser.add_argument('indices',
                    help='indices file name (e.g. top_conv1.txt)')
parser.add_argument('--arch', '-a', choices=models.archs.keys(), default='nin',
                    help='Convnet architecture')
parser.add_argument('--root', '-R', default='.',
                    help='Root directory path of image files')
parser.add_argument('--out', '-o', default='result',
                    help='Output directory')
parser.add_argument('--categories', '-c', default='categories.txt',
                    help='Path to category list file')


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
