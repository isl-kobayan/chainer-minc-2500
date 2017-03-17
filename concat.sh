DATA_ROOT=./minc-2500
TEACH_DATA=./minc-2500/all_list.txt
SRC=concat_image.py
VGG_BEST_DIR=./result/vgg16/20160923-1544_bs21

#python ${SRC} -t 9 -c 3 -p 4 -s 5 --withoutframe -R ${VGG_BEST_DIR}/conv2_2
#python ${SRC} -t 9 -c 3 -p 5 -s 2 --withoutframe -R ${VGG_BEST_DIR}/conv3_3
#python ${SRC} -t 9 -c 3 -p 4 -s 1 --withoutframe -R ${VGG_BEST_DIR}/conv4_3
#python ${SRC} -t 9 -c 3 -p 4 -s 1 --withoutframe -R ${VGG_BEST_DIR}/conv5_3

#python ${SRC} -t 9 -c 3 -p 5 -s 1 --withoutframe -R ${VGG_BEST_DIR}/fc6
#python ${SRC} -t 9 -c 3 -p 5 -s 1 --withoutframe -R ${VGG_BEST_DIR}/fc7

python ${SRC} -t 9 -c 3 -p 4 -s 5 --withoutframe -R ${VGG_BEST_DIR}/deconv_nobias_030/deconv_conv2_2
python ${SRC} -t 9 -c 3 -p 5 -s 2 --withoutframe -R ${VGG_BEST_DIR}/deconv_nobias_030/deconv_conv3_3
python ${SRC} -t 9 -c 3 -p 4 -s 1 --withoutframe -R ${VGG_BEST_DIR}/deconv_nobias_030/deconv_conv4_3
python ${SRC} -t 9 -c 3 -p 4 -s 1 --withoutframe -R ${VGG_BEST_DIR}/deconv_nobias_030/deconv_conv5_3

python ${SRC} -t 9 -c 3 -p 5 -s 1 --withoutframe -R ${VGG_BEST_DIR}/deconv_nobias_030/deconv_fc6
python ${SRC} -t 9 -c 3 -p 5 -s 1 --withoutframe -R ${VGG_BEST_DIR}/deconv_nobias_030/deconv_fc7
