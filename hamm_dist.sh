SRC=hamm_dist.py
n=100
BASE_DIR=./result/vgg16/best/features

python ${SRC} -g 0 -n $n -p xor ${BASE_DIR}/fc6_0.npy ${BASE_DIR}/fc7_0.npy ${BASE_DIR}/h67_0_xor.npy
python ${SRC} -g 0 -n $n -p xor ${BASE_DIR}/fc6_1.npy ${BASE_DIR}/fc7_1.npy ${BASE_DIR}/h67_1_xor.npy
python ${SRC} -g 0 -n $n -p xor ${BASE_DIR}/fc6_5.npy ${BASE_DIR}/fc7_5.npy ${BASE_DIR}/h67_5_xor.npy

python ${SRC} -g 0 -n $n -p xor ${BASE_DIR}/fc6_0.npy ${BASE_DIR}/pool5_0.npy ${BASE_DIR}/h65_0_xor.npy
python ${SRC} -g 0 -n $n -p xor ${BASE_DIR}/fc6_1.npy ${BASE_DIR}/pool5_1.npy ${BASE_DIR}/h65_1_xor.npy
python ${SRC} -g 0 -n $n -p xor ${BASE_DIR}/fc6_5.npy ${BASE_DIR}/pool5_5.npy ${BASE_DIR}/h65_5_xor.npy

python ${SRC} -g 0 -n $n -p xor ${BASE_DIR}/fc7_0.npy ${BASE_DIR}/pool5_0.npy ${BASE_DIR}/h75_0_xor.npy
python ${SRC} -g 0 -n $n -p xor ${BASE_DIR}/fc7_1.npy ${BASE_DIR}/pool5_1.npy ${BASE_DIR}/h75_1_xor.npy
python ${SRC} -g 0 -n $n -p xor ${BASE_DIR}/fc7_5.npy ${BASE_DIR}/pool5_5.npy ${BASE_DIR}/h75_5_xor.npy

python ${SRC} -g 0 -n $n -p nand ${BASE_DIR}/fc6_0.npy ${BASE_DIR}/fc7_0.npy ${BASE_DIR}/h67_0_nand.npy
python ${SRC} -g 0 -n $n -p nand ${BASE_DIR}/fc6_1.npy ${BASE_DIR}/fc7_1.npy ${BASE_DIR}/h67_1_nand.npy
python ${SRC} -g 0 -n $n -p nand ${BASE_DIR}/fc6_5.npy ${BASE_DIR}/fc7_5.npy ${BASE_DIR}/h67_5_nand.npy

python ${SRC} -g 0 -n $n -p nand ${BASE_DIR}/fc6_0.npy ${BASE_DIR}/pool5_0.npy ${BASE_DIR}/h65_0_nand.npy
python ${SRC} -g 0 -n $n -p nand ${BASE_DIR}/fc6_1.npy ${BASE_DIR}/pool5_1.npy ${BASE_DIR}/h65_1_nand.npy
python ${SRC} -g 0 -n $n -p nand ${BASE_DIR}/fc6_5.npy ${BASE_DIR}/pool5_5.npy ${BASE_DIR}/h65_5_nand.npy

python ${SRC} -g 0 -n $n -p nand ${BASE_DIR}/fc7_0.npy ${BASE_DIR}/pool5_0.npy ${BASE_DIR}/h75_0_nand.npy
python ${SRC} -g 0 -n $n -p nand ${BASE_DIR}/fc7_1.npy ${BASE_DIR}/pool5_1.npy ${BASE_DIR}/h75_1_nand.npy
python ${SRC} -g 0 -n $n -p nand ${BASE_DIR}/fc7_5.npy ${BASE_DIR}/pool5_5.npy ${BASE_DIR}/h75_5_nand.npy
