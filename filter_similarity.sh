DATA_ROOT=./minc-2500
TEACH_DATA=./minc-2500/all_list.txt
SRC=filter_similarity.py

#python ${SRC} -a vgg16 --initmodel ./result/vgg16/20160923-1544_bs21/model -b 20 -l conv1_2
#python ${SRC} -a vgg16 --initmodel ./result/vgg16/20160923-1544_bs21/model -b 20 -l conv2_2
#python ${SRC} -a vgg16 --initmodel ./result/vgg16/20160923-1544_bs21/model -b 20 -l conv3_3
#python ${SRC} -a vgg16 --initmodel ./result/vgg16/20160923-1544_bs21/model -b 20 -l conv4_3
#python ${SRC} -a vgg16 --initmodel ./result/vgg16/20160923-1544_bs21/model -b 20 -l conv5_3
#python ${SRC} -a vgg16 --initmodel ./result/vgg16/20160923-1544_bs21/model -b 20 -l fc6
#python ${SRC} -a vgg16 --initmodel ./result/vgg16/20160923-1544_bs21/model -b 20 -l fc7
#python ${SRC} -a vgg16 --initmodel ./result/vgg16/20160923-1544_bs21/model -b 20 -l fc8

python ${SRC} -a vgg16 --initmodel ./result/vgg16/20160923-1544_bs21/model -d euclidean -b 100 -l conv1_2
python ${SRC} -a vgg16 --initmodel ./result/vgg16/20160923-1544_bs21/model -d euclidean -b 100 -l conv2_2
python ${SRC} -a vgg16 --initmodel ./result/vgg16/20160923-1544_bs21/model -d euclidean -b 100 -l conv3_3
python ${SRC} -a vgg16 --initmodel ./result/vgg16/20160923-1544_bs21/model -d euclidean -b 100 -l conv4_3
python ${SRC} -a vgg16 --initmodel ./result/vgg16/20160923-1544_bs21/model -d euclidean -b 100 -l conv5_3
python ${SRC} -a vgg16 --initmodel ./result/vgg16/20160923-1544_bs21/model -d euclidean -b 100 -l fc6
python ${SRC} -a vgg16 --initmodel ./result/vgg16/20160923-1544_bs21/model -d euclidean -b 100 -l fc7
python ${SRC} -a vgg16 --initmodel ./result/vgg16/20160923-1544_bs21/model -d euclidean -b 100 -l fc8

aplay ~/ミュージック/se_maoudamashii_system24.wav
