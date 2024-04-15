
#!/usr/bin/env bash

DATASET=$1
FRAME_PATH=$2
OUT_LIST_PATH=$3

python tools/build_file_list.py ${DATASET} ${FRAME_PATH} ${OUT_LIST_PATH} --shuffle
