DATA_ROOT=./minc-2500
TEACH_DATA=./minc-2500/all_list.txt
SRC=extract_features.py

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

#python ${SRC} -a vgg16 --initmodel ./result/vgg16/20160923-1544_bs21/model -b 20 -g 0 -R ${DATA_ROOT} ${TEACH_DATA} -l conv1_2
#python ${SRC} -a vgg16 --initmodel ./result/vgg16/20160923-1544_bs21/model -b 20 -g 0 -R ${DATA_ROOT} ${TEACH_DATA} -l conv2_2
#python ${SRC} -a vgg16 --initmodel ./result/vgg16/20160923-1544_bs21/model -b 20 -g 0 -R ${DATA_ROOT} ${TEACH_DATA} -l conv3_3
#python ${SRC} -a vgg16 --initmodel ./result/vgg16/20160923-1544_bs21/model -b 20 -g 0 -R ${DATA_ROOT} ${TEACH_DATA} -l conv4_3
#python ${SRC} -a vgg16 --initmodel ./result/vgg16/20160923-1544_bs21/model -b 20 -g 0 -R ${DATA_ROOT} ${TEACH_DATA} -l conv5_3
#python ${SRC} -a vgg16 --initmodel ./result/vgg16/20160923-1544_bs21/model -b 20 -g 0 -R ${DATA_ROOT} ${TEACH_DATA} -l fc6 --operation None
python ${SRC} -a vgg16 --initmodel ./result/vgg16/20160923-1544_bs21/model -b 20 -g 0 -R ${DATA_ROOT} ${TEACH_DATA} -l fc7 --operation None
python ${SRC} -a vgg16 --initmodel ./result/vgg16/20160923-1544_bs21/model -b 20 -g 0 -R ${DATA_ROOT} ${TEACH_DATA} -l fc8 --operation None

#python ${SRC} -a googlenet --initmodel model_googlenet -b 50 -l conv1 -v ./ilsvrc_val.txt -r /home/isl-ws16/Caffe/data/ilsvrc12/resize/val -o .
#python ${SRC} -a googlenet --initmodel model_googlenet -b 50 -l conv2_reduce -v ./ilsvrc_val.txt -r /home/isl-ws16/Caffe/data/ilsvrc12/resize/val -o .
#python ${SRC} -a googlenet --initmodel model_googlenet -b 50 -l conv2 -v ./ilsvrc_val.txt -r /home/isl-ws16/Caffe/data/ilsvrc12/resize/val -o .
#python ${SRC} -a googlenet --initmodel model_googlenet -b 50 -l inception_3a -v ./ilsvrc_val.txt -r /home/isl-ws16/Caffe/data/ilsvrc12/resize/val -o .
#python ${SRC} -a googlenet --initmodel model_googlenet -b 50 -l inception_3b -v ./ilsvrc_val.txt -r /home/isl-ws16/Caffe/data/ilsvrc12/resize/val -o .
#python ${SRC} -a googlenet --initmodel model_googlenet -b 50 -l inception_4a -v ./ilsvrc_val.txt -r /home/isl-ws16/Caffe/data/ilsvrc12/resize/val -o .
#python ${SRC} -a googlenet --initmodel model_googlenet -b 50 -l inception_4b -v ./ilsvrc_val.txt -r /home/isl-ws16/Caffe/data/ilsvrc12/resize/val -o .
#python ${SRC} -a googlenet --initmodel model_googlenet -b 50 -l inception_4c -v ./ilsvrc_val.txt -r /home/isl-ws16/Caffe/data/ilsvrc12/resize/val -o .
#python ${SRC} -a googlenet --initmodel model_googlenet -b 50 -l inception_4d -v ./ilsvrc_val.txt -r /home/isl-ws16/Caffe/data/ilsvrc12/resize/val -o .
#python ${SRC} -a googlenet --initmodel model_googlenet -b 50 -l inception_4e -v ./ilsvrc_val.txt -r /home/isl-ws16/Caffe/data/ilsvrc12/resize/val -o .
#python ${SRC} -a googlenet --initmodel model_googlenet -b 50 -l inception_5a -v ./ilsvrc_val.txt -r /home/isl-ws16/Caffe/data/ilsvrc12/resize/val -o .
#python ${SRC} -a googlenet --initmodel model_googlenet -b 50 -l inception_5b -v ./ilsvrc_val.txt -r /home/isl-ws16/Caffe/data/ilsvrc12/resize/val -o .

#python ${SRC} -a alex --initmodel model_alex -b 50 -l conv1 -v ./ilsvrc_val.txt -r /home/isl-ws16/Caffe/data/ilsvrc12/resize/val -o .
#python ${SRC} -a alex --initmodel model_alex -b 50 -l conv2 -v ./ilsvrc_val.txt -r /home/isl-ws16/Caffe/data/ilsvrc12/resize/val -o .
#python ${SRC} -a alex --initmodel model_alex -b 50 -l conv3 -v ./ilsvrc_val.txt -r /home/isl-ws16/Caffe/data/ilsvrc12/resize/val -o .
#python ${SRC} -a alex --initmodel model_alex -b 50 -l conv4 -v ./ilsvrc_val.txt -r /home/isl-ws16/Caffe/data/ilsvrc12/resize/val -o .
#python ${SRC} -a alex --initmodel model_alex -b 50 -l conv5 -v ./ilsvrc_val.txt -r /home/isl-ws16/Caffe/data/ilsvrc12/resize/val -o .

#python ${SRC} -a alex --initmodel model_alex -b 50 -l conv1 -v ./fmd_val.txt -o .
#python ${SRC} -a alex --initmodel model_alex -b 50 -l conv2 -v ./fmd_val.txt -o .
#python ${SRC} -a alex --initmodel model_alex -b 50 -l conv3 -v ./fmd_val.txt -o .
#python ${SRC} -a alex --initmodel model_alex -b 50 -l conv4 -v ./fmd_val.txt -o .
#python ${SRC} -a alex --initmodel model_alex -b 50 -l conv5 -v ./fmd_val.txt -o .

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
#python ${SRC} -a googlenet --initmodel ./result/googlenet/20160922-0749_bs23/model -b 50 -g 0 -R ${DATA_ROOT} ${TEACH_DATA} -l loss3_classifier

aplay ~/ミュージック/se_maoudamashii_system24.wav
