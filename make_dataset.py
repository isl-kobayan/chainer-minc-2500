#!/usr/bin/python
# -*- coding: utf-8 -*-
import argparse
import random

parser = argparse.ArgumentParser(
    description='Make teacher-data as space-separated text file.')
parser.add_argument('file', help='Path to image list file')
parser.add_argument('categories', help='Path to image list file')
parser.add_argument('out', help='Path to output image-label list file')
parser.add_argument('--shuffle', '-s', default=False, action='store_true',
                    help='shuffles lists if this flag is set (default: False)')
parser.add_argument('--depth', '-d', type=int, default=1, help='depth of category name directory')
args = parser.parse_args()

with open(args.categories) as fc:
    categories = [c.rstrip() for c in fc.readlines()]
    cat2num = dict(zip(categories, [str(i) for i in range(len(categories))]))

    with open(args.out, 'w') as fo:
        with open(args.file) as f:
            lines = f.readlines()
            if args.shuffle:
                random.shuffle(lines)
            for line in lines:
                dirs = line.split("/")
                fo.write(line.rstrip() + '\t' + cat2num[dirs[args.depth]] + '\n')
