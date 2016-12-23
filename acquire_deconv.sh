DATA_ROOT=./minc-2500
TEACH_DATA=./minc-2500/all_list.txt
SRC=acquire_deconv.py

#python ${SRC} -a alex --finetune -b 50 -g 0 -R ${DATA_ROOT} ${TEACH_DATA} -m ilsvrc_2012_mean.npy -l conv1
#python ${SRC} -a alex --finetune -b 50 -g 0 -R ${DATA_ROOT} ${TEACH_DATA} -m ilsvrc_2012_mean.npy -l conv2
#python ${SRC} -a alex --finetune -b 50 -g 0 -R ${DATA_ROOT} ${TEACH_DATA} -m ilsvrc_2012_mean.npy -l conv3
#python ${SRC} -a alex --finetune -b 50 -g 0 -R ${DATA_ROOT} ${TEACH_DATA} -m ilsvrc_2012_mean.npy -l conv4
#python ${SRC} -a alex --finetune -b 50 -g 0 -R ${DATA_ROOT} ${TEACH_DATA} -m ilsvrc_2012_mean.npy -l conv5
#python ${SRC} -a alex --finetune -b 50 -g 0 -R ${DATA_ROOT} ${TEACH_DATA} -m ilsvrc_2012_mean.npy -l pool1
#python ${SRC} -a alex --finetune -b 50 -g 0 -R ${DATA_ROOT} ${TEACH_DATA} -m ilsvrc_2012_mean.npy -l pool2
#python ${SRC} -a alex --finetune -b 50 -g 0 -R ${DATA_ROOT} ${TEACH_DATA} -m ilsvrc_2012_mean.npy -l pool5
#python ${SRC} -a alex --finetune -b 50 -g 0 -R ${DATA_ROOT} ${TEACH_DATA} -m ilsvrc_2012_mean.npy -l norm1
#python ${SRC} -a alex --finetune -b 50 -g 0 -R ${DATA_ROOT} ${TEACH_DATA} -m ilsvrc_2012_mean.npy -l norm2
#python ${SRC} -a alex --finetune -b 50 -g 0 -R ${DATA_ROOT} ${TEACH_DATA} -m ilsvrc_2012_mean.npy -l fc6
#python ${SRC} -a alex --finetune -b 50 -g 0 -R ${DATA_ROOT} ${TEACH_DATA} -m ilsvrc_2012_mean.npy -l fc7
#python ${SRC} -a alex --finetune -b 50 -g 0 -R ${DATA_ROOT} ${TEACH_DATA} -m ilsvrc_2012_mean.npy -l fc8

#python ${SRC} -a minc-alex --finetune -b 50 -g 0 -R ${DATA_ROOT} ${TEACH_DATA} -l conv1
#python ${SRC} -a minc-alex --finetune -b 50 -g 0 -R ${DATA_ROOT} ${TEACH_DATA} -l conv2
#python ${SRC} -a minc-alex --finetune -b 50 -g 0 -R ${DATA_ROOT} ${TEACH_DATA} -l conv3
#python ${SRC} -a minc-alex --finetune -b 50 -g 0 -R ${DATA_ROOT} ${TEACH_DATA} -l conv4
#python ${SRC} -a minc-alex --finetune -b 50 -g 0 -R ${DATA_ROOT} ${TEACH_DATA} -l conv5

#python ${SRC} -a googlenet --finetune -b 50 -g 0 -R ${DATA_ROOT} ${TEACH_DATA} -l conv1
#python ${SRC} -a googlenet --finetune -b 50 -g 0 -R ${DATA_ROOT} ${TEACH_DATA} -l conv2_reduce
#python ${SRC} -a googlenet --finetune -b 50 -g 0 -R ${DATA_ROOT} ${TEACH_DATA} -l conv2
#python ${SRC} -a googlenet --finetune -b 50 -g 0 -R ${DATA_ROOT} ${TEACH_DATA} -l inception_3a
#python ${SRC} -a googlenet --finetune -b 50 -g 0 -R ${DATA_ROOT} ${TEACH_DATA} -l inception_3b
#python ${SRC} -a googlenet --finetune -b 50 -g 0 -R ${DATA_ROOT} ${TEACH_DATA} -l inception_4a
#python ${SRC} -a googlenet --finetune -b 50 -g 0 -R ${DATA_ROOT} ${TEACH_DATA} -l inception_4b
#python ${SRC} -a googlenet --finetune -b 50 -g 0 -R ${DATA_ROOT} ${TEACH_DATA} -l inception_4c
#python ${SRC} -a googlenet --finetune -b 50 -g 0 -R ${DATA_ROOT} ${TEACH_DATA} -l inception_4d
#python ${SRC} -a googlenet --finetune -b 50 -g 0 -R ${DATA_ROOT} ${TEACH_DATA} -l inception_4e
#python ${SRC} -a googlenet --finetune -b 50 -g 0 -R ${DATA_ROOT} ${TEACH_DATA} -l inception_5a
#python ${SRC} -a googlenet --finetune -b 50 -g 0 -R ${DATA_ROOT} ${TEACH_DATA} -l inception_5b

#python ${SRC} -a minc-googlenet --finetune -b 50 -g 0 -R ${DATA_ROOT} ${TEACH_DATA} -l conv1
#python ${SRC} -a minc-googlenet --finetune -b 50 -g 0 -R ${DATA_ROOT} ${TEACH_DATA} -l conv2_reduce
#python ${SRC} -a minc-googlenet --finetune -b 50 -g 0 -R ${DATA_ROOT} ${TEACH_DATA} -l conv2
#python ${SRC} -a minc-googlenet --finetune -b 50 -g 0 -R ${DATA_ROOT} ${TEACH_DATA} -l inception_3a
#python ${SRC} -a minc-googlenet --finetune -b 50 -g 0 -R ${DATA_ROOT} ${TEACH_DATA} -l inception_3b
#python ${SRC} -a minc-googlenet --finetune -b 50 -g 0 -R ${DATA_ROOT} ${TEACH_DATA} -l inception_4a
#python ${SRC} -a minc-googlenet --finetune -b 50 -g 0 -R ${DATA_ROOT} ${TEACH_DATA} -l inception_4b
#python ${SRC} -a minc-googlenet --finetune -b 50 -g 0 -R ${DATA_ROOT} ${TEACH_DATA} -l inception_4c
#python ${SRC} -a minc-googlenet --finetune -b 50 -g 0 -R ${DATA_ROOT} ${TEACH_DATA} -l inception_4d
#python ${SRC} -a minc-googlenet --finetune -b 50 -g 0 -R ${DATA_ROOT} ${TEACH_DATA} -l inception_4e
#python ${SRC} -a minc-googlenet --finetune -b 50 -g 0 -R ${DATA_ROOT} ${TEACH_DATA} -l inception_5a
#python ${SRC} -a minc-googlenet --finetune -b 50 -g 0 -R ${DATA_ROOT} ${TEACH_DATA} -l inception_5b


#python ${SRC} -a vgg16 --finetune -b 20 -g 0 -R ${DATA_ROOT} ${TEACH_DATA} -l conv1_2
#python ${SRC} -a vgg16 --finetune -b 20 -g 0 -R ${DATA_ROOT} ${TEACH_DATA} -l conv2_2
#python ${SRC} -a vgg16 --finetune -b 20 -g 0 -R ${DATA_ROOT} ${TEACH_DATA} -l conv3_3
#python ${SRC} -a vgg16 --finetune -b 20 -g 0 -R ${DATA_ROOT} ${TEACH_DATA} -l conv4_3
#python ${SRC} -a vgg16 --finetune -b 20 -g 0 -R ${DATA_ROOT} ${TEACH_DATA} -l conv5_3

#python ${SRC} -a minc-vgg16 --finetune -b 20 -g 0 -R ${DATA_ROOT} ${TEACH_DATA} -l conv1_2
#python ${SRC} -a minc-vgg16 --finetune -b 20 -g 0 -R ${DATA_ROOT} ${TEACH_DATA} -l conv2_2
#python ${SRC} -a minc-vgg16 --finetune -b 20 -g 0 -R ${DATA_ROOT} ${TEACH_DATA} -l conv3_3
#python ${SRC} -a minc-vgg16 --finetune -b 20 -g 0 -R ${DATA_ROOT} ${TEACH_DATA} -l conv4_3
#python ${SRC} -a minc-vgg16 --finetune -b 20 -g 0 -R ${DATA_ROOT} ${TEACH_DATA} -l conv5_3

python ${SRC} -a vgg16 --initmodel ./result/vgg16/20160923-1544_bs21/model -b 20 -g 0 -R ${DATA_ROOT} ${TEACH_DATA} -l conv1_2
python ${SRC} -a vgg16 --initmodel ./result/vgg16/20160923-1544_bs21/model -b 20 -g 0 -R ${DATA_ROOT} ${TEACH_DATA} -l conv2_2
#python ${SRC} -a vgg16 --initmodel ./result/vgg16/20160923-1544_bs21/model -b 20 -g 0 -R ${DATA_ROOT} ${TEACH_DATA} -l conv3_3
#python ${SRC} -a vgg16 --initmodel ./result/vgg16/20160923-1544_bs21/model -b 20 -g 0 -R ${DATA_ROOT} ${TEACH_DATA} -l conv4_3
#python ${SRC} -a vgg16 --initmodel ./result/vgg16/20160923-1544_bs21/model -b 20 -g 0 -R ${DATA_ROOT} ${TEACH_DATA} -l conv5_3

#python ${SRC} -a googlenet --initmodel ./result/googlenet/20160922-0749_bs23/model -b 50 -g 0 -R ${DATA_ROOT} ${TEACH_DATA} -l conv1
#python ${SRC} -a googlenet --initmodel ./result/googlenet/20160922-0749_bs23/model -b 50 -g 0 -R ${DATA_ROOT} ${TEACH_DATA} -l conv2_reduce
#python ${SRC} -a googlenet --initmodel ./result/googlenet/20160922-0749_bs23/model -b 50 -g 0 -R ${DATA_ROOT} ${TEACH_DATA} -l conv2
#python ${SRC} -a googlenet --initmodel ./result/googlenet/20160922-0749_bs23/model -b 50 -g 0 -R ${DATA_ROOT} ${TEACH_DATA} -l inception_3a
#python ${SRC} -a googlenet --initmodel ./result/googlenet/20160922-0749_bs23/model -b 50 -g 0 -R ${DATA_ROOT} ${TEACH_DATA} -l inception_3b
#python ${SRC} -a googlenet --initmodel ./result/googlenet/20160922-0749_bs23/model -b 50 -g 0 -R ${DATA_ROOT} ${TEACH_DATA} -l inception_4a
#python ${SRC} -a googlenet --initmodel ./result/googlenet/20160922-0749_bs23/model -b 50 -g 0 -R ${DATA_ROOT} ${TEACH_DATA} -l inception_4b
#python ${SRC} -a googlenet --initmodel ./result/googlenet/20160922-0749_bs23/model -b 50 -g 0 -R ${DATA_ROOT} ${TEACH_DATA} -l inception_4c
#python ${SRC} -a googlenet --initmodel ./result/googlenet/20160922-0749_bs23/model -b 50 -g 0 -R ${DATA_ROOT} ${TEACH_DATA} -l inception_4d
#python ${SRC} -a googlenet --initmodel ./result/googlenet/20160922-0749_bs23/model -b 50 -g 0 -R ${DATA_ROOT} ${TEACH_DATA} -l inception_4e
#python ${SRC} -a googlenet --initmodel ./result/googlenet/20160922-0749_bs23/model -b 50 -g 0 -R ${DATA_ROOT} ${TEACH_DATA} -l inception_5a
#python ${SRC} -a googlenet --initmodel ./result/googlenet/20160922-0749_bs23/model -b 50 -g 0 -R ${DATA_ROOT} ${TEACH_DATA} -l inception_5b
#python ${SRC} -a googlenet --initmodel ./result/googlenet/20160922-0749_bs23/model -b 50 -g 0 -R ${DATA_ROOT} ${TEACH_DATA} -l pool5


aplay ~/ミュージック/se_maoudamashii_system24.wav
