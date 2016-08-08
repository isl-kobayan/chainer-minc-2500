import subprocess
import json
import argparse
import itertools
import models
import train_minc2500

parser = argparse.ArgumentParser(
    description='Learning convnet from MINC-2500 dataset')
parser.add_argument('train', help='Path to training image-label list file')
parser.add_argument('val', help='Path to validation image-label list file')
parser.add_argument('--arch', '-a', choices=models.archs.keys(), default='nin',
                    help='Convnet architecture')
parser.add_argument('--batchsize', '-B', type=int, default=32,
                    help='Learning minibatch size')
parser.add_argument('--baselr', default=0.001, type=float,
                    help='Base learning rate')
parser.add_argument('--gamma', default=0.7, type=float,
                    help='Base learning rate')
parser.add_argument('--epoch', '-E', type=int, default=10,
                    help='Number of epochs to train')
parser.add_argument('--gpu', '-g', type=int, default=-1,
                    help='GPU ID (negative value indicates CPU')
parser.add_argument('--finetune', '-f', default=False, action='store_true',
                    help='do fine-tuning if this flag is set (default: False)')
parser.add_argument('--initmodel',
                    help='Initialize the model from given file')
parser.add_argument('--loaderjob', '-j', type=int,
                    help='Number of parallel data loading processes')
parser.add_argument('--mean', '-m', default='mean.npy',
                    help='Mean file (computed by compute_mean.py)')
parser.add_argument('--resume', '-r', default='',
                    help='Initialize the trainer from given file')
parser.add_argument('--out', '-o', default='result',
                    help='Output directory')
parser.add_argument('--root', '-R', default='.',
                    help='Root directory path of image files')
parser.add_argument('--val_batchsize', '-b', type=int, default=50,
                    help='Validation minibatch size')
parser.add_argument('--test', action='store_true')
parser.set_defaults(test=False)
args = parser.parse_args()

archs = ['vgg16', 'vgg19']
batchsizes = [23, 32]
baselrs = [0.0005, 0.001, 0.005]
gammas = [0.9, 0.5]

#archs = ['googlenet']
#batchsizes = [32]
#baselrs = [0.001]
#gammas = [0.5]

path_gridsearchlog = args.out + '/gridsearch.log'
results="dir\tarch\tbatchsize\tbaselr\tgamma\tloss\taccuracy\n"
with open(path_gridsearchlog, 'a') as f:
    f.write(results)

for arch, bs, baselr, gamma in itertools.product(archs, batchsizes, baselrs, gammas):
    args.arch = arch
    args.batchsize = bs
    args.baselr = baselr
    args.gamma = gamma
    val_result = train_minc2500.main(args)

    result = val_result['outputdir'] + '\t' \
        + arch + '\t' + str(bs) + '\t' + str(baselr) + '\t' + str(gamma) + '\t' \
        + str(val_result['validation/main/loss']) + '\t' + str(val_result['validation/main/accuracy'])
    with open(path_gridsearchlog, 'a') as f:
        f.write(result + '\n')
    results += result + '\n'

print(results)
