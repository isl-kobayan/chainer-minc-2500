# chainer-minc-2500
train/validate minc-2500 dataset with [chainer 1.15](https://github.com/pfnet/chainer)

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
* if using fine-tuning, download mean file and imagenet-pretrained model:
```
python download_mean_file.py
```
```
python download_model.py
```
* available pre-trained architechures are:
  * [AlexNet](https://github.com/BVLC/caffe/tree/master/models/bvlc_alexnet)
  * [GoogLeNet](https://github.com/BVLC/caffe/tree/master/models/bvlc_googlenet)
  * VGG ([16 layers](https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-readme-md) and [19 layers](https://gist.github.com/ksimonyan/3785162f95cd2d5fee77#file-readme-md))
  * [NIN (Network-in-Network)](https://gist.github.com/mavenlin/d802a5849de39225bcc6)
  * [SqueezeNet (ver. 1.0 and ver. 1.1)](https://github.com/DeepScale/SqueezeNet)  

  see [Caffe Model Zoo](https://github.com/BVLC/caffe/wiki/Model-Zoo) for more details.

### train/validate
example:

* architecture: GoogLeNet
* fine-tuning: on
* use GPU
```
python train_minc2500.py ./minc-2500/shuffled_labels/train1.txt \
./minc-2500/shuffled_labels/validate1.txt -a googlenet --finetune -E 10 -R ./minc-2500 -g 0
```

#### grid search
```
python gridsearch.py ./minc-2500/shuffled_labels/train1.txt \
./minc-2500/shuffled_labels/validate1.txt --finetune -E 1 -R ./minc-2500 -g 0
```
