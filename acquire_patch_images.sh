DATA_ROOT=./minc-2500
TEACH_DATA=./minc-2500/all_list.txt
SRC=acquire_patch_images.py

#python ${SRC} -a alex -b 50 -R ${DATA_ROOT} ${TEACH_DATA} -l conv1
#python ${SRC} -a alex -b 50 -R ${DATA_ROOT} ${TEACH_DATA} -l conv2
#python ${SRC} -a alex -b 50 -R ${DATA_ROOT} ${TEACH_DATA} -l conv3
#python ${SRC} -a alex -b 50 -R ${DATA_ROOT} ${TEACH_DATA} -l conv4
#python ${SRC} -a alex -b 50 -R ${DATA_ROOT} ${TEACH_DATA} -l conv5
#python ${SRC} -a alex -b 50 -R ${DATA_ROOT} ${TEACH_DATA} -l pool1
#python ${SRC} -a alex -b 50 -R ${DATA_ROOT} ${TEACH_DATA} -l pool2
#python ${SRC} -a alex -b 50 -R ${DATA_ROOT} ${TEACH_DATA} -l pool5
#python ${SRC} -a alex -b 50 -R ${DATA_ROOT} ${TEACH_DATA} -l norm1
#python ${SRC} -a alex -b 50 -R ${DATA_ROOT} ${TEACH_DATA} -l norm2
#python ${SRC} -a alex -b 50 -R ${DATA_ROOT} ${TEACH_DATA} -l fc6
#python ${SRC} -a alex -b 50 -R ${DATA_ROOT} ${TEACH_DATA} -l fc7
#python ${SRC} -a alex -b 50 -R ${DATA_ROOT} ${TEACH_DATA} -l fc8

#python ${SRC} -a minc-alex -b 50 -R ${DATA_ROOT} ${TEACH_DATA} -l conv1
#python ${SRC} -a minc-alex -b 50 -R ${DATA_ROOT} ${TEACH_DATA} -l conv2
#python ${SRC} -a minc-alex -b 50 -R ${DATA_ROOT} ${TEACH_DATA} -l conv3
#python ${SRC} -a minc-alex -b 50 -R ${DATA_ROOT} ${TEACH_DATA} -l conv4
#python ${SRC} -a minc-alex -b 50 -R ${DATA_ROOT} ${TEACH_DATA} -l conv5

#python ${SRC} -a googlenet -b 50 -R ${DATA_ROOT} ${TEACH_DATA} -l conv1
#python ${SRC} -a googlenet -b 50 -R ${DATA_ROOT} ${TEACH_DATA} -l conv2_reduce
#python ${SRC} -a googlenet -b 50 -R ${DATA_ROOT} ${TEACH_DATA} -l conv2
#python ${SRC} -a googlenet -b 50 -R ${DATA_ROOT} ${TEACH_DATA} -l inception_3a
#python ${SRC} -a googlenet -b 50 -R ${DATA_ROOT} ${TEACH_DATA} -l inception_3b
#python ${SRC} -a googlenet -b 50 -R ${DATA_ROOT} ${TEACH_DATA} -l inception_4a
#python ${SRC} -a googlenet -b 50 -R ${DATA_ROOT} ${TEACH_DATA} -l inception_4b
#python ${SRC} -a googlenet -b 50 -R ${DATA_ROOT} ${TEACH_DATA} -l inception_4c
#python ${SRC} -a googlenet -b 50 -R ${DATA_ROOT} ${TEACH_DATA} -l inception_4d
#python ${SRC} -a googlenet -b 50 -R ${DATA_ROOT} ${TEACH_DATA} -l inception_4e
#python ${SRC} -a googlenet -b 50 -R ${DATA_ROOT} ${TEACH_DATA} -l inception_5a
#python ${SRC} -a googlenet -b 50 -R ${DATA_ROOT} ${TEACH_DATA} -l inception_5b

INIT_DIR=./result/googlenet/20160922-0749_bs23

python ${SRC} -a googlenet -b 50 -R ${DATA_ROOT} ${TEACH_DATA} --initdir ${INIT_DIR} -l conv1
python ${SRC} -a googlenet -b 50 -R ${DATA_ROOT} ${TEACH_DATA} --initdir ${INIT_DIR} -l conv2_reduce
python ${SRC} -a googlenet -b 50 -R ${DATA_ROOT} ${TEACH_DATA} --initdir ${INIT_DIR} -l conv2
python ${SRC} -a googlenet -b 50 -R ${DATA_ROOT} ${TEACH_DATA} --initdir ${INIT_DIR} -l inception_3a
python ${SRC} -a googlenet -b 50 -R ${DATA_ROOT} ${TEACH_DATA} --initdir ${INIT_DIR} -l inception_3b
python ${SRC} -a googlenet -b 50 -R ${DATA_ROOT} ${TEACH_DATA} --initdir ${INIT_DIR} -l inception_4a
python ${SRC} -a googlenet -b 50 -R ${DATA_ROOT} ${TEACH_DATA} --initdir ${INIT_DIR} -l inception_4b
python ${SRC} -a googlenet -b 50 -R ${DATA_ROOT} ${TEACH_DATA} --initdir ${INIT_DIR} -l inception_4c
python ${SRC} -a googlenet -b 50 -R ${DATA_ROOT} ${TEACH_DATA} --initdir ${INIT_DIR} -l inception_4d
python ${SRC} -a googlenet -b 50 -R ${DATA_ROOT} ${TEACH_DATA} --initdir ${INIT_DIR} -l inception_4e
python ${SRC} -a googlenet -b 50 -R ${DATA_ROOT} ${TEACH_DATA} --initdir ${INIT_DIR} -l inception_5a
python ${SRC} -a googlenet -b 50 -R ${DATA_ROOT} ${TEACH_DATA} --initdir ${INIT_DIR} -l inception_5b

#python ${SRC} -a minc-googlenet -b 50 -R ${DATA_ROOT} ${TEACH_DATA} -l conv1
#python ${SRC} -a minc-googlenet -b 50 -R ${DATA_ROOT} ${TEACH_DATA} -l conv2_reduce
#python ${SRC} -a minc-googlenet -b 50 -R ${DATA_ROOT} ${TEACH_DATA} -l conv2
#python ${SRC} -a minc-googlenet -b 50 -R ${DATA_ROOT} ${TEACH_DATA} -l inception_3a
#python ${SRC} -a minc-googlenet -b 50 -R ${DATA_ROOT} ${TEACH_DATA} -l inception_3b
#python ${SRC} -a minc-googlenet -b 50 -R ${DATA_ROOT} ${TEACH_DATA} -l inception_4a
#python ${SRC} -a minc-googlenet -b 50 -R ${DATA_ROOT} ${TEACH_DATA} -l inception_4b
#python ${SRC} -a minc-googlenet -b 50 -R ${DATA_ROOT} ${TEACH_DATA} -l inception_4c
#python ${SRC} -a minc-googlenet -b 50 -R ${DATA_ROOT} ${TEACH_DATA} -l inception_4d
#python ${SRC} -a minc-googlenet -b 50 -R ${DATA_ROOT} ${TEACH_DATA} -l inception_4e
#python ${SRC} -a minc-googlenet -b 50 -R ${DATA_ROOT} ${TEACH_DATA} -l inception_5a
#python ${SRC} -a minc-googlenet -b 50 -R ${DATA_ROOT} ${TEACH_DATA} -l inception_5b


#python ${SRC} -a vgg16 -b 20 -R ${DATA_ROOT} ${TEACH_DATA} -l conv1_2
#python ${SRC} -a vgg16 -b 20 -R ${DATA_ROOT} ${TEACH_DATA} -l conv2_2
#python ${SRC} -a vgg16 -b 20 -R ${DATA_ROOT} ${TEACH_DATA} -l conv3_3
#python ${SRC} -a vgg16 -b 20 -R ${DATA_ROOT} ${TEACH_DATA} -l conv4_3
#python ${SRC} -a vgg16 -b 20 -R ${DATA_ROOT} ${TEACH_DATA} -l conv5_3

#python ${SRC} -a minc-vgg16 -b 20 -R ${DATA_ROOT} ${TEACH_DATA} -l conv1_2
#python ${SRC} -a minc-vgg16 -b 20 -R ${DATA_ROOT} ${TEACH_DATA} -l conv2_2
#python ${SRC} -a minc-vgg16 -b 20 -R ${DATA_ROOT} ${TEACH_DATA} -l conv3_3
#python ${SRC} -a minc-vgg16 -b 20 -R ${DATA_ROOT} ${TEACH_DATA} -l conv4_3
#python ${SRC} -a minc-vgg16 -b 20 -R ${DATA_ROOT} ${TEACH_DATA} -l conv5_3

#python ${SRC} -a vgg16 -b 20 -R ${DATA_ROOT} ${TEACH_DATA} -l conv1_2 --initdir ./result/vgg16/20160923-1544_bs21
#python ${SRC} -a vgg16 -b 20 -R ${DATA_ROOT} ${TEACH_DATA} -l conv2_2 --initdir ./result/vgg16/20160923-1544_bs21
#python ${SRC} -a vgg16 -b 20 -R ${DATA_ROOT} ${TEACH_DATA} -l conv3_3 --initdir ./result/vgg16/20160923-1544_bs21
#python ${SRC} -a vgg16 -b 20 -R ${DATA_ROOT} ${TEACH_DATA} -l conv4_3 --initdir ./result/vgg16/20160923-1544_bs21
#python ${SRC} -a vgg16 -b 20 -R ${DATA_ROOT} ${TEACH_DATA} -l conv5_3 --initdir ./result/vgg16/20160923-1544_bs21
#python ${SRC} -a vgg16 -b 20 -R ${DATA_ROOT} ${TEACH_DATA} -l fc6 --initdir ./result/vgg16/20160923-1544_bs21 --nobounds
#python ${SRC} -a vgg16 -b 20 -R ${DATA_ROOT} ${TEACH_DATA} -l fc7 --initdir ./result/vgg16/20160923-1544_bs21 --nobounds
#python ${SRC} -a vgg16 -b 20 -R ${DATA_ROOT} ${TEACH_DATA} -l fc8 --initdir ./result/vgg16/20160923-1544_bs21 --nobounds

aplay ~/ミュージック/se_maoudamashii_system24.wav
