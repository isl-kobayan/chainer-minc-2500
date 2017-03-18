# chainer-minc-2500
train/validate minc-2500 dataset in [chainer](https://github.com/pfnet/chainer)

MINC-2500 dataset: [download](http://opensurfaces.cs.cornell.edu/static/minc/minc-2500.tar.gz) 

## Article
**[Material Recognition in the Wild with the Materials in Context Database](http://opensurfaces.cs.cornell.edu/publications/minc/)**  
Sean Bell and Paul Upchurch and Noah Snavely and Kavita Bala,  
Computer Vision and Pattern Recognition (CVPR), 2015.

## Requirements
* chainer 1.15+
* numpy
* scikit-learn
* Pillow
* matplotlib
* tqdm
* dominate

## Usage
### preparation
* put ''minc-2500'' directory in ''chainer-minc-2500'' directory
* make image-label dataset before training

  ```
  python make_dataset.py ./minc-2500 shuffled_labels --shuffle
  python make_all_list.py
  ```
* if using fine-tuning, download mean file and imagenet-pretrained model

  ```
  python download_mean_file.py
  ```
  
  ```
  python download_model.py
  ```
* available pre-trained architechures
  * [AlexNet](https://github.com/BVLC/caffe/tree/master/models/bvlc_alexnet)
  * [GoogLeNet(v1)](https://github.com/BVLC/caffe/tree/master/models/bvlc_googlenet)
  * VGG ([16 layers](https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-readme-md) and [19 layers](https://gist.github.com/ksimonyan/3785162f95cd2d5fee77#file-readme-md))
  * [NIN (Network-in-Network)](https://gist.github.com/mavenlin/d802a5849de39225bcc6)
  * [SqueezeNet (ver. 1.0 and ver. 1.1)](https://github.com/DeepScale/SqueezeNet)  

  see [Caffe Model Zoo](https://github.com/BVLC/caffe/wiki/Model-Zoo) for more details.

### train/validate
example: train(fine-tune) GoogLeNet with GPU
```
python train_minc2500.py ./minc-2500/shuffled_labels/train1.txt \
./minc-2500/shuffled_labels/validate1.txt -a googlenet --finetune -E 10 -R ./minc-2500 -g 0
```

#### random search & grid search
```
python randomsearch.py ./minc-2500/shuffled_labels/train1.txt \
./minc-2500/shuffled_labels/validate1.txt --finetune -E 1 -R ./minc-2500 -g 0 -t 20
```

```
python gridsearch.py ./minc-2500/shuffled_labels/train1.txt \
./minc-2500/shuffled_labels/validate1.txt --finetune -E 1 -R ./minc-2500 -g 0
```

### filter visualization (畳み込みフィルタ及び全結合層のニューロンの可視化)
example: visualize "conv1" layer of imagenet-pretrained AlexNet (bvlc_alexnet.caffemodel) using all minc-2500 images

1. extract filter output  
 このプログラムは、指定した層の各フィルタ（全結合層の場合は各ニューロン）を刺激する上位9枚(-t 9)の画像を調べます。  
  
  ```
  python extract_features.py ./minc-2500/all_list.txt -a alex --finetune -b 50 -g 0 -R ./minc-2500 \
  -m ilsvrc_2012_mean.npy -l conv1 -t 9
  ```  
  This program generates "./result/alex/extract/features/top_conv1.txt".

2. acquire most activated image patch  
 このプログラムは、手順1の結果をもとに、活性化パッチ画像を取得します。  
  
  ```
  python acquire_patches.py ./minc-2500/all_list.txt -a alex --finetune -b 50 -g 0 -R ./minc-2500 \
  -m ilsvrc_2012_mean.npy -l conv1
  ```  
  Please execute procedure 1 before executing this procedure.  
  This program generates images (./result/alex/extract/conv1/*.png).  
  This program also generates "./result/alex/extract/features/maxbounds_conv1.txt" and "./result/alex/extract/features/maxloc_conv1.txt".

3. acquire deconv image  
 このプログラムは、手順1の結果をもとに、Zeiler & Fergusの手法を用いて可視化画像を取得します。  
 （このプログラムを実行する前に、手順2を実行する必要はありません）  
  
  ```
  python acquire_patches.py ./minc-2500/all_list.txt -a alex --finetune -b 50 -g 0 -R ./minc-2500 \
  -m ilsvrc_2012_mean.npy -l conv1
  ```  
  Please execute procedure 1 before executing this procedure (You don't have to execute procedure 2).  
  This program generates images (./result/alex/extract/deconv_conv1/*.png).  

You can concatenate images
