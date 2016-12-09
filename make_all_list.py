#!/usr/bin/env python2

import glob
import os.path
import sys

if __name__=='__main__':
  categories=[x.strip() for x in open('./minc-2500/categories.txt').readlines()]
  all_list=[]
  with open('./minc-2500/all_list.txt', 'w') as f:
    for i,x in enumerate(categories):
      for j,y in enumerate(sorted(glob.glob('./minc-2500/images/{}/*'.format(x)))):
        all_list.append('images/{}/{}\t{}\n'.format(x, os.path.basename(y), i))
    f.writelines(all_list)
