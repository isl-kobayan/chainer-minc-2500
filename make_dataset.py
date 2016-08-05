#!/usr/bin/python
# -*- coding: utf-8 -*-
import argparse
import random
import os

def convert_dataset(filepath, output_path, categories_path):
    with open(categories_path) as fc:
        categories = [c.rstrip() for c in fc.readlines()]
        cat2num = dict(zip(categories, [str(i) for i in range(len(categories))]))
        print(categories_path)

        with open(output_path, 'w') as fo:
            print(output_path)
            with open(filepath) as f:
                print(filepath)
                lines = f.readlines()
                if args.shuffle:
                    random.shuffle(lines)
                for line in lines:
                    dirs = line.split("/")
                    fo.write(line.rstrip() + '\t' + cat2num[dirs[args.depth]] + '\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Make teacher-data as space-separated text file.')
    parser.add_argument('path', default='./minc-2500',
        help='Path to minc-2500 dataset directory')
    parser.add_argument('out', default='label_list',
        help='Path to output image-label list directory')
    #parser.add_argument('file', help='Path to image list file')
    #parser.add_argument('categories', help='Path to image list file')
    #parser.add_argument('out', help='Path to output image-label list file')
    parser.add_argument('--shuffle', '-s', default=False, action='store_true',
                        help='shuffles lists if this flag is set (default: False)')
    parser.add_argument('--depth', '-d', type=int, default=1, help='depth of category name directory')
    args = parser.parse_args()

    categories_path = os.path.join(args.path, 'categories.txt')
    output_dir = os.path.join(args.path, args.out)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    filetypes = ['train', 'validate', 'test']
    for filetype in filetypes:
        for i in range(1, 6):
            filename = os.path.join(args.path, 'labels', filetype + str(i) + '.txt')
            output_path = os.path.join(output_dir, filetype + str(i) + '.txt')
            convert_dataset(filename, output_path, categories_path)
