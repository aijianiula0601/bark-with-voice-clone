
#!/bin/bash

set -ex

curdir=$(pwd)
echo "curdir:$curdir"
cd "$curdir" || exit

base_dir='/mnt/cephfs/hjh/train_record/tts/bark_with_voice_clone/train_data'

cd $base_dir

split -n l/64 train.txt part


for file in part*
do
  echo "处理文件: $file"
  python ${curdir}/process_part.py $base_dir $file &

done

wait

rm -rf part*