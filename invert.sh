SRC=invert.py
VGG_BEST_MODEL=./result/vgg16/best/model
n_categories=23
for ((i=0; i<${n_categories}; i++)); do
    python ${SRC} -g 0 -a vgg16 --initmodel ${VGG_BEST_MODEL} $i -o ./result/img-test7 -i 2500 --lambda_a 1 --lambda_tv 1 --lambda_lp 10
done
