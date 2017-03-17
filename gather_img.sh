#!/bin/bash
# show_acts.pyを先に実行し、comp.logを作成しておくこと。
# chainer-FMDの認識結果ログ(comp.log)を元に、画像を次のように保存します。
# ・正しく認識できた画像をcorrectディレクトリに
# ・誤認識した画像をincorrectディレクトリに
# ・認識結果をconfusionディレクトリに
# confusionディレクトリには、正解クラスごとにディレクトリが作成され、
# 例えば、fabricの画像をfoliageに誤認識した場合、foliage_fabric_**.jpgのようなファイル名で保存します。

#usage: gather_ox.sh ****/comp.log

# remove file if exists
function rm_if_exists() {
  if [ -e $1 ]; then
    if [ -f $1 ] ; then
      rm $1
    elif [ -d $1 ] ; then
      rm -r $1
    fi
  fi
}

function mkdir_if_not_exists() {
  if [ ! -e $1 ]; then
    mkdir $1
  fi
}

function gather_image_from_list() {
  if [ -e $2 ]; then
    for line in `cat $1`; do
      #echo "${line}"
      dir=${line%/*}
      mtl=${dir##*/}
      fname=${line##*/}
      mkdir_if_not_exists "$2/${mtl}"
      cp $line "$2/${mtl}/${fname}"
    done
  fi
}

function gather_image_from_list2() {
  OLDIFS=$IFS
  #ntl=( `cat $3` )
  if [ -e $2 ]; then
    IFS=$'\n'
    for line in `awk 'NR>1 {print}' $1`; do
      #echo "${line}"
      IFS=$'\t '
      arr=( `echo ${line}` )
      #echo ${arr[0]}
      #echo ${arr[1]}
      #echo ${arr[2]}

      filepath=${arr[0]}
      #pred=${ntl[${arr[2]}]}
      pred=${arr[2]}
      dir=${filepath%/*}
      mtl=${dir##*/}
      fname=${filepath##*/}
      mkdir_if_not_exists "${2}/${mtl}"
      cp $filepath "${2}/${mtl}/${pred}_${fname}"
    done
  fi
  IFS=$OLDIFS
}

function gather_incorrect() {
  OLDIFS=$IFS
  #ntl=( `cat $3` )
  if [ -e $2 ]; then
    IFS=$'\n'
    for line in `awk 'NR>1 {print}' $1`; do
      #echo "${line}"
      IFS=$'\t '
      arr=( `echo ${line}` )
      #echo ${arr[0]}
      #echo ${arr[1]}
      #echo ${arr[2]}

      filepath=${arr[0]}
      #pred=${ntl[${arr[2]}]}
      truth=${arr[1]}
      pred=${arr[2]}

      dir=${filepath%/*}
      mtl=${dir##*/}
      fname=${filepath##*/}
      if [[ $pred != $truth ]]; then
        mkdir_if_not_exists "${2}/${mtl}"
        cp $filepath "${2}/${mtl}/${pred}_${fname}"
      fi
    done
  fi
  IFS=$OLDIFS
}

filespath="./files.txt"
labelspath="./labels.txt"
imagelistpath="./image_list.txt"
num2labelpath="./num2label.txt"

#in_dir="./image"
if [ $# -lt 1 ]; then
  echo "指定された引数は$#個です。" 1>&2
  echo "実行するには1個の引数が必要です。" 1>&2
  exit 1
fi

in_dir=${1%/*}
correct_dir="${in_dir}/crop"
incorrect_dir="${in_dir}/incorrect"
confusion_dir="${in_dir}/confusion"
correct_list="${in_dir}/correct.txt"
incorrect_list="${in_dir}/incorrect.txt"

rm_if_exists $correct_list
rm_if_exists $incorrect_list
rm_if_exists $correct_dir
rm_if_exists $incorrect_dir
rm_if_exists $confusion_dir

# 認識結果のファイルから、正解した画像のファイル名をcorrect_listに、
# 誤認識した画像のファイル名をincorrect_listに記録
# ファイルの最初の行は読み飛ばす
awk '{ if ( NR > 1 && $2 == $3 ) {print $1} }' $1 > $correct_list
awk '{ if ( NR > 1 && $2 != $3 ) {print $1} }' $1 > $incorrect_list

mkdir $correct_dir
mkdir $incorrect_dir
mkdir $confusion_dir

gather_image_from_list $correct_list $correct_dir
#gather_image_from_list $incorrect_list $incorrect_dir
gather_incorrect $1 $incorrect_dir
gather_image_from_list2 $1 $confusion_dir
