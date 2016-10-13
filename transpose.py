#!/usr/bin/env python
""" Transpose tsv file """

import argparse
import numpy as np
import pandas as pd
import os

def main(args):
    data = pd.read_csv(args.infile, header=None, delimiter=args.delim)
    if args.out is None:
        args.out = os.path.join(os.path.dirname(args.infile),
            't_' + os.path.basename(args.infile))
    data.T.to_csv(args.out, sep=args.delim, header=None, index=False)

parser = argparse.ArgumentParser(
    description='transpose tsv file')
parser.add_argument('infile', help='input tsv file')
parser.add_argument('--out', '-o', help='output tsv file name')
parser.add_argument('--delim', '-d', default='\t',
                    help='delimiter')

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
