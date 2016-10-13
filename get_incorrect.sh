DATA_ROOT=./minc-2500
TEACH_DATA=./minc-2500/shuffled_labels/test1.txt
SRC=get_incorrect.py
CATEGORIES=./minc-2500/categories.txt

python ${SRC} -a vgg16 -R ${DATA_ROOT} ${TEACH_DATA} --categories ${CATEGORIES} --initdir ./result/vgg16/20160923-1544_bs21

aplay ~/ミュージック/se_maoudamashii_system24.wav
