#!/bin/bash

cd `dirname $0`
source ../.env
echo "HOST_DATA_DIR: $HOST_DATA_DIR"

mkdir -p $HOST_DATA_DIR/coco
cd $HOST_DATA_DIR/coco

urls=(
    "http://images.cocodataset.org/zips/train2017.zip"
    "http://images.cocodataset.org/zips/val2017.zip"
    "http://images.cocodataset.org/zips/test2017.zip"
    "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
)

# 各ファイルをバックグラウンドでダウンロード・展開
for url in "${urls[@]}"; do
    filename=$(basename "$url")
    
    (
        curl -o "$filename" "$url" &&
        unzip "$filename" &&
        rm "$filename"
    ) &
done

# 全てのバックグラウンドジョブの終了を待つ
wait
