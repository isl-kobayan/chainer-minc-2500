n=$1
declare -i offset
offset=${#1}
offset=${offset}+1
args=${*:$offset}

python train_minc2500.py ./minc-2500/shuffled_labels/train$n.txt ./minc-2500/shuffled_labels/validate$n.txt -R ./minc-2500 $args
