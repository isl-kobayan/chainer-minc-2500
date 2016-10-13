n=$1
declare -i offset
offset=${#1}
offset=${offset}+1
args=${*:$offset}

python extract_features.py ./minc-2500/shuffled_labels/validate$n.txt -R ./minc-2500 $args
