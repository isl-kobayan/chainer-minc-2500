#python get_activated_patches.py -a alex --finetune -b 10 -l conv1 -n ./alex/acts_conv1.npy -v ./ilsvrc_val.txt -r /home/isl-ws16/Caffe/data/ilsvrc12/resize/val -o .
#python get_activated_patches.py -a alex --finetune -b 10 -l conv2 -n ./alex/acts_conv2.npy -v ./ilsvrc_val.txt -r /home/isl-ws16/Caffe/data/ilsvrc12/resize/val -o .
#python get_activated_patches.py -a alex --finetune -b 10 -l conv3 -n ./alex/acts_conv3.npy -v ./ilsvrc_val.txt -r /home/isl-ws16/Caffe/data/ilsvrc12/resize/val -o .
#python get_activated_patches.py -a alex --finetune -b 10 -l conv4 -n ./alex/acts_conv4.npy -v ./ilsvrc_val.txt -r /home/isl-ws16/Caffe/data/ilsvrc12/resize/val -o .
#python get_activated_patches.py -a alex --finetune -b 10 -l conv5 -n ./alex/acts_conv5.npy -v ./ilsvrc_val.txt -r /home/isl-ws16/Caffe/data/ilsvrc12/resize/val -o .
#python get_activated_patches.py -a alex --finetune -b 10 -l pool1 -n ./alex/acts_pool1.npy -v ./ilsvrc_val.txt -r /home/isl-ws16/Caffe/data/ilsvrc12/resize/val -o .
#python get_activated_patches.py -a alex --finetune -b 10 -l pool2 -n ./alex/acts_pool2.npy -v ./ilsvrc_val.txt -r /home/isl-ws16/Caffe/data/ilsvrc12/resize/val -o .
#python get_activated_patches.py -a alex --finetune -b 10 -l pool5 -n ./alex/acts_pool5.npy -v ./ilsvrc_val.txt -r /home/isl-ws16/Caffe/data/ilsvrc12/resize/val -o .
#python get_activated_patches.py -a alex --finetune -b 10 -l relu1 -n ./alex/acts_relu1.npy -v ./ilsvrc_val.txt -r /home/isl-ws16/Caffe/data/ilsvrc12/resize/val -o .
#python get_activated_patches.py -a alex --finetune -b 10 -l relu2 -n ./alex/acts_relu2.npy -v ./ilsvrc_val.txt -r /home/isl-ws16/Caffe/data/ilsvrc12/resize/val -o .
#python get_activated_patches.py -a alex --finetune -b 10 -l relu3 -n ./alex/acts_relu3.npy -v ./ilsvrc_val.txt -r /home/isl-ws16/Caffe/data/ilsvrc12/resize/val -o .
#python get_activated_patches.py -a alex --finetune -b 10 -l relu4 -n ./alex/acts_relu4.npy -v ./ilsvrc_val.txt -r /home/isl-ws16/Caffe/data/ilsvrc12/resize/val -o .
#python get_activated_patches.py -a alex --finetune -b 10 -l relu5 -n ./alex/acts_relu5.npy -v ./ilsvrc_val.txt -r /home/isl-ws16/Caffe/data/ilsvrc12/resize/val -o .

#python get_activated_patches.py -a googlenet --finetune -b 10 -l conv1 -n ./googlenet/acts_conv1.npy -v ./ilsvrc_val.txt -r /home/isl-ws16/Caffe/data/ilsvrc12/resize/val -o .
#python get_activated_patches.py -a googlenet --finetune -b 10 -l conv2_reduce -n ./googlenet/acts_conv2_reduce.npy -v ./ilsvrc_val.txt -r /home/isl-ws16/Caffe/data/ilsvrc12/resize/val -o .
#python get_activated_patches.py -a googlenet --finetune -b 10 -l conv2 -n ./googlenet/acts_conv2.npy -v ./ilsvrc_val.txt -r /home/isl-ws16/Caffe/data/ilsvrc12/resize/val -o .
#python get_activated_patches.py -a googlenet --finetune -b 10 -l inception_3a -n ./googlenet/acts_inception_3a.npy -v ./ilsvrc_val.txt -r /home/isl-ws16/Caffe/data/ilsvrc12/resize/val -o .
#python get_activated_patches.py -a googlenet --finetune -b 10 -l inception_3b -n ./googlenet/acts_inception_3b.npy -v ./ilsvrc_val.txt -r /home/isl-ws16/Caffe/data/ilsvrc12/resize/val -o .
#python get_activated_patches.py -a googlenet --finetune -b 10 -l inception_4a -n ./googlenet/acts_inception_4a.npy -v ./ilsvrc_val.txt -r /home/isl-ws16/Caffe/data/ilsvrc12/resize/val -o .
#python get_activated_patches.py -a googlenet --finetune -b 10 -l inception_4b -n ./googlenet/acts_inception_4b.npy -v ./ilsvrc_val.txt -r /home/isl-ws16/Caffe/data/ilsvrc12/resize/val -o .
#python get_activated_patches.py -a googlenet --finetune -b 10 -l inception_4c -n ./googlenet/acts_inception_4c.npy -v ./ilsvrc_val.txt -r /home/isl-ws16/Caffe/data/ilsvrc12/resize/val -o .
#python get_activated_patches.py -a googlenet --finetune -b 10 -l inception_4d -n ./googlenet/acts_inception_4d.npy -v ./ilsvrc_val.txt -r /home/isl-ws16/Caffe/data/ilsvrc12/resize/val -o .
#python get_activated_patches.py -a googlenet --finetune -b 10 -l inception_4e -n ./googlenet/acts_inception_4e.npy -v ./ilsvrc_val.txt -r /home/isl-ws16/Caffe/data/ilsvrc12/resize/val -o .
#python get_activated_patches.py -a googlenet --finetune -b 10 -l inception_5a -n ./googlenet/acts_inception_5a.npy -v ./ilsvrc_val.txt -r /home/isl-ws16/Caffe/data/ilsvrc12/resize/val -o .
#python get_activated_patches.py -a googlenet --finetune -b 10 -l inception_5b -n ./googlenet/acts_inception_5b.npy -v ./ilsvrc_val.txt -r /home/isl-ws16/Caffe/data/ilsvrc12/resize/val -o .

#python get_activated_patches.py -a googlenet --initmodel model_googlenet -b 10 -l conv1 -n ./googlenet/acts_conv1.npy -v ./ilsvrc_val.txt -r /home/isl-ws16/Caffe/data/ilsvrc12/resize/val -o .
#python get_activated_patches.py -a googlenet --initmodel model_googlenet -b 10 -l conv2_reduce -n ./googlenet/acts_conv2_reduce.npy -v ./ilsvrc_val.txt -r /home/isl-ws16/Caffe/data/ilsvrc12/resize/val -o .
#python get_activated_patches.py -a googlenet --initmodel model_googlenet -b 10 -l conv2 -n ./googlenet/acts_conv2.npy -v ./ilsvrc_val.txt -r /home/isl-ws16/Caffe/data/ilsvrc12/resize/val -o .
#python get_activated_patches.py -a googlenet --initmodel model_googlenet -b 10 -l inception_3a -n ./googlenet/acts_inception_3a.npy -v ./ilsvrc_val.txt -r /home/isl-ws16/Caffe/data/ilsvrc12/resize/val -o .
#python get_activated_patches.py -a googlenet --initmodel model_googlenet -b 10 -l inception_3b -n ./googlenet/acts_inception_3b.npy -v ./ilsvrc_val.txt -r /home/isl-ws16/Caffe/data/ilsvrc12/resize/val -o .
#python get_activated_patches.py -a googlenet --initmodel model_googlenet -b 10 -l inception_4a -n ./googlenet/acts_inception_4a.npy -v ./ilsvrc_val.txt -r /home/isl-ws16/Caffe/data/ilsvrc12/resize/val -o .
#python get_activated_patches.py -a googlenet --initmodel model_googlenet -b 10 -l inception_4b -n ./googlenet/acts_inception_4b.npy -v ./ilsvrc_val.txt -r /home/isl-ws16/Caffe/data/ilsvrc12/resize/val -o .
#python get_activated_patches.py -a googlenet --initmodel model_googlenet -b 10 -l inception_4c -n ./googlenet/acts_inception_4c.npy -v ./ilsvrc_val.txt -r /home/isl-ws16/Caffe/data/ilsvrc12/resize/val -o .
#python get_activated_patches.py -a googlenet --initmodel model_googlenet -b 10 -l inception_4d -n ./googlenet/acts_inception_4d.npy -v ./ilsvrc_val.txt -r /home/isl-ws16/Caffe/data/ilsvrc12/resize/val -o .
#python get_activated_patches.py -a googlenet --initmodel model_googlenet -b 10 -l inception_4e -n ./googlenet/acts_inception_4e.npy -v ./ilsvrc_val.txt -r /home/isl-ws16/Caffe/data/ilsvrc12/resize/val -o .
#python get_activated_patches.py -a googlenet --initmodel model_googlenet -b 10 -l inception_5a -n ./googlenet/acts_inception_5a.npy -v ./ilsvrc_val.txt -r /home/isl-ws16/Caffe/data/ilsvrc12/resize/val -o .
#python get_activated_patches.py -a googlenet --initmodel model_googlenet -b 10 -l inception_5b -n ./googlenet/acts_inception_5b.npy -v ./ilsvrc_val.txt -r /home/isl-ws16/Caffe/data/ilsvrc12/resize/val -o .

#python get_activated_patches.py -a alex --initmodel model_alex -b 10 -l conv1 -n ./alex/acts_conv1.npy -v ./ilsvrc_val.txt -r /home/isl-ws16/Caffe/data/ilsvrc12/resize/val -o .
#python get_activated_patches.py -a alex --initmodel model_alex -b 10 -l conv2 -n ./alex/acts_conv2.npy -v ./ilsvrc_val.txt -r /home/isl-ws16/Caffe/data/ilsvrc12/resize/val -o .
#python get_activated_patches.py -a alex --initmodel model_alex -b 10 -l conv3 -n ./alex/acts_conv3.npy -v ./ilsvrc_val.txt -r /home/isl-ws16/Caffe/data/ilsvrc12/resize/val -o .
#python get_activated_patches.py -a alex --initmodel model_alex -b 10 -l conv4 -n ./alex/acts_conv4.npy -v ./ilsvrc_val.txt -r /home/isl-ws16/Caffe/data/ilsvrc12/resize/val -o .
#python get_activated_patches.py -a alex --initmodel model_alex -b 10 -l conv5 -n ./alex/acts_conv5.npy -v ./ilsvrc_val.txt -r /home/isl-ws16/Caffe/data/ilsvrc12/resize/val -o .

python get_activated_patches.py -a alex --initmodel model_alex -b 10 -l conv1 -n ./alex/acts_conv1.npy -v ./fmd_val.txt -o .
python get_activated_patches.py -a alex --initmodel model_alex -b 10 -l conv2 -n ./alex/acts_conv2.npy -v ./fmd_val.txt -o .
python get_activated_patches.py -a alex --initmodel model_alex -b 10 -l conv3 -n ./alex/acts_conv3.npy -v ./fmd_val.txt -o .
python get_activated_patches.py -a alex --initmodel model_alex -b 10 -l conv4 -n ./alex/acts_conv4.npy -v ./fmd_val.txt -o .
python get_activated_patches.py -a alex --initmodel model_alex -b 10 -l conv5 -n ./alex/acts_conv5.npy -v ./fmd_val.txt -o .

python get_activated_patches.py -a googlenet --initmodel model_googlenet -b 10 -l conv1        -n ./googlenet/acts_conv1.npy        -v ./fmd_val.txt -o .
python get_activated_patches.py -a googlenet --initmodel model_googlenet -b 10 -l conv2_reduce -n ./googlenet/acts_conv2_reduce.npy -v ./fmd_val.txt -o .
python get_activated_patches.py -a googlenet --initmodel model_googlenet -b 10 -l conv2        -n ./googlenet/acts_conv2.npy        -v ./fmd_val.txt -o .
python get_activated_patches.py -a googlenet --initmodel model_googlenet -b 10 -l inception_3a -n ./googlenet/acts_inception_3a.npy -v ./fmd_val.txt -o .
python get_activated_patches.py -a googlenet --initmodel model_googlenet -b 10 -l inception_3b -n ./googlenet/acts_inception_3b.npy -v ./fmd_val.txt -o .
python get_activated_patches.py -a googlenet --initmodel model_googlenet -b 10 -l inception_4a -n ./googlenet/acts_inception_4a.npy -v ./fmd_val.txt -o .
python get_activated_patches.py -a googlenet --initmodel model_googlenet -b 10 -l inception_4b -n ./googlenet/acts_inception_4b.npy -v ./fmd_val.txt -o .
python get_activated_patches.py -a googlenet --initmodel model_googlenet -b 10 -l inception_4c -n ./googlenet/acts_inception_4c.npy -v ./fmd_val.txt -o .
python get_activated_patches.py -a googlenet --initmodel model_googlenet -b 10 -l inception_4d -n ./googlenet/acts_inception_4d.npy -v ./fmd_val.txt -o .
python get_activated_patches.py -a googlenet --initmodel model_googlenet -b 10 -l inception_4e -n ./googlenet/acts_inception_4e.npy -v ./fmd_val.txt -o .
python get_activated_patches.py -a googlenet --initmodel model_googlenet -b 10 -l inception_5a -n ./googlenet/acts_inception_5a.npy -v ./fmd_val.txt -o .
python get_activated_patches.py -a googlenet --initmodel model_googlenet -b 10 -l inception_5b -n ./googlenet/acts_inception_5b.npy -v ./fmd_val.txt -o .
