import subprocess
import json
import argparse
import itertools
import models
import train_minc2500
import random

parser = train_minc2500.parser
parser.add_argument('--trial', '-t', type=int, default=20,
                    help='trial count')
parser.set_defaults(finetune=True)
parser.set_defaults(gpu=0)

args = parser.parse_args()

#archs = ['googlenet', 'vgg16']
batchsize_range = 20, 32
baselr_range = 0.0001, 0.005
gamma_range = 0.3, 0.9
momentum_range = 0.9, 1.0

archs = ['googlenetbn']
#batchsizes = [32]
#baselrs = [0.001]
#gammas = [0.5]

path_randomsearchlog = args.out + '/randomsearch.log'
results="dir\tarch\tbatchsize\tbaselr\tgamma\tmomentum\tloss\taccuracy\n"
with open(path_randomsearchlog, 'a') as f:
    f.write(results)

for i in range(args.trial):
    args.arch = random.choice(archs)
    args.batchsize = random.randint(*batchsize_range)
    args.baselr = random.uniform(*baselr_range)
    args.gamma = random.uniform(*gamma_range)
    args.momentum = random.uniform(*momentum_range)
    val_result = train_minc2500.main(args)

    result = "{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\t{7}".format(val_result['outputdir'],
        args.arch, args.batchsize, args.baselr, args.gamma, args.momentum,
        val_result['validation/main/loss'], val_result['validation/main/accuracy'])
    with open(path_randomsearchlog, 'a') as f:
        f.write(result + '\n')
    results += result + '\n'

print(results)
