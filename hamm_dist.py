#!/usr/bin/env python
""" Calculate hamming distance between 2 npy file """

import argparse
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import chainer
from chainer import cuda
import cupy

def main(args):
    x1 = np.load(args.infile1)
    x2 = np.load(args.infile2)
    assert len(x1.shape) == 2, 'infile1 should be 2d array!'
    assert len(x2.shape) == 2, 'infile2 should be 2d array!'
    assert x1.shape[0] == x2.shape[0], 'two infile should have same rows!'
    x1 = np.unpackbits(x1, axis=1)
    x2 = np.unpackbits(x2, axis=1)
    r1 = x1.shape[1] if args.row1 == 0 else args.row1
    r2 = x2.shape[1] if args.row2 == 0 else args.row2
    N = x1.shape[0]
    print(r1, r2, N)
    x1 = np.packbits(x1[:, :r1].T, axis=1)
    x2 = np.packbits(x2[:, :r2].T, axis=1)

    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        x1 = cuda.to_gpu(x1)
        x2 = cuda.to_gpu(x2)
        xp = cupy
    else:
        xp = np
    # popcount LUT
    pc = xp.zeros(256, dtype=np.uint8)
    for i in range(256):
        pc[i] = ( i & 1 ) + pc[i/2]

    hamm = xp.zeros((r1, r2), dtype=np.int32)
    for i in tqdm(range(r1)):
        x1i = xp.tile(x1[i], (r2, 1))
        if args.operation == 'xor':
            hamm[i] = xp.take(pc, xp.bitwise_xor(x1i, x2).astype(np.int32)).sum(axis=1)
        elif args.operation == 'nand':
            hamm[i] = xp.take(pc, xp.invert(xp.bitwise_and(x1i, x2)).astype(np.int32)).sum(axis=1)
        #for j in range(r2):
            #hamm[i, j] = xp.take(pc, xp.bitwise_xor(x1[i], x2[j])).sum()
    x1non0 = xp.tile((x1.sum(axis=1)>0), (r2, 1)).T.astype(np.int32)
    x2non0 = xp.tile((x2.sum(axis=1)>0), (r1, 1)).astype(np.int32)
    print(x1non0.shape, x2non0.shape)
    non0filter = x1non0 * x2non0
    print(non0filter.max(), non0filter.min())
    hamm = non0filter * hamm + np.iinfo(np.int32).max * (1 - non0filter)
    #non0filter *= np.iinfo(np.int32).max
    #hamm *= non0filter
    if xp == cupy:
        hamm = hamm.get()
    #xp.savetxt(args.out, hamm, delimiter=args.delim)
    np.save(args.out, hamm)

    if args.nearest > 0:
        hamm_s = np.sort(hamm.flatten())
        hamm_as = np.argsort(hamm.flatten())
        x, y = np.unravel_index(hamm_as[:args.nearest], hamm.shape)
        fname, ext = os.path.splitext(args.out)
        np.savetxt(fname + '_top{0}.tsv'.format(args.nearest),
            np.concatenate((x[np.newaxis], y[np.newaxis], hamm_s[np.newaxis,:args.nearest]), axis=0).T,
            fmt='%d', delimiter='\t')


parser = argparse.ArgumentParser(
    description='transpose tsv file')
parser.add_argument('infile1', help='input npy file')
parser.add_argument('infile2', help='input npy file')
parser.add_argument('--row1', type=int, default=0, help='rows of infile1')
parser.add_argument('--row2', type=int, default=0, help='rows of infile2')
parser.add_argument('--nearest', '-n', type=int, default=0, help='output the N nearest pairs')
parser.add_argument('--gpu', '-g', type=int, default=-1, help='gpu number')
parser.add_argument('--operation', '-p', choices=('xor', 'nand'), default='xor', help='operation')
parser.add_argument('out', help='output npy file name')
parser.add_argument('--delim', '-d', default='\t',
                    help='delimiter')

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
