# chainer-minc-2500
train/validate minc-2500 dataset

MINC-2500 dataset: [download](http://opensurfaces.cs.cornell.edu/static/minc/minc-2500.tar.gz) 

## Article
**[Material Recognition in the Wild with the Materials in Context Database](http://opensurfaces.cs.cornell.edu/publications/minc/)**  
Sean Bell and Paul Upchurch and Noah Snavely and Kavita Bala,  
Computer Vision and Pattern Recognition (CVPR), 2015.

## Usage
### preparation
* put ''minc-2500'' directory in ''chainer-minc-2500'' directory
* make image-label dataset before training:
```
python make_dataset.py ./minc-2500 shuffled_labels --shuffle
```
### train/validate
* architecture: GoogLeNet
* fine-tuning: on
* use GPU
```
python train_minc2500.py ./minc-2500/shuffled_labels/train1.txt \
./minc-2500/shuffled_labels/validate1.txt -a googlenet --finetune -E 10 -R ./minc-2500 -g 0
```
