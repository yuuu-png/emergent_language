#!/bin/bash
# これは今回作成したプログラムがシード値に依存するかどうかを調べるためのスクリプトです。

cd `dirname $0`/..

for i in {1..10}
do
    make egg GPU=0 ARGS="--seed=$i --name=seed_$i"
done