import argparse
import os
import sys

sys.path.append('.')
from pyActionRecog import parse_directory, build_split_list
from pyActionRecog import parse_split_file


parser = argparse.ArgumentParser()

parser.add_argument('dataset', type=str, choices=['ucfcrime', 'hmdb51', 'activitynet_1.2', 'activitynet_1.3'])
parser.add_argument('frame_path', type=str, help="root directory holding the frames")
parser.add_argument('out_list_path', type=str, help='output directory path to hold the splits file')
parser.add_argument('--num_split', type=int, default=3)
parser.add_argument('--shuffle', action='store_true', default=False)

args = parser.parse_args()

dataset = args.dataset
frame_path = args.frame_path
out_path = args.out_list_path
num_split = args.num_split
shuffle = args.shuffle

# operation
print('processing dataset {}'.format(dataset))
split_tp = parse_split_file(dataset)
f_info = parse_directory(frame_path)

print('writing list files for training/testing')
for i in range(max(num_split, len(split_tp))):
    lists = build_split_list(split_tp, f_info, i, shuffle)
    #print(lists[0])
    open(os.path.join(out_path, '{}_rgb_train_split_{}.txt'.format(dataset, i+1)), 'w').writelines(lists[0])
    open(os.path.join(out_path, '{}_rgb_val_split_{}.txt'.format(dataset, i+1)), 'w').writelines(lists[1])
