#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
import glob
import sys
import cv2
import argparse

out_path = ''

count = 0

def dump_frames(vid_path, num_of_videos):
    video = cv2.VideoCapture(vid_path)
    vid_name = vid_path.split('/')[-1].split('.')[0]
    out_full_path = os.path.join(out_path, vid_name)

    fcount = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    try:
        os.mkdir(out_full_path)
    except OSError:
        pass
    file_list = []
    for i in range(fcount):
        ret, frame = video.read()
        if not ret:
            break
        resizedFrame = cv2.resize(frame, (64, 64))
        cv2.imwrite('{}/img_{:05d}.jpg'.format(out_full_path, i+1), resizedFrame)
        access_path = '{}/img_{:05d}.jpg'.format(out_full_path, i+1)
        file_list.append(access_path)

    global count
    count += 1
    print('--> {}/{} -> {} done'.format(count, num_of_videos, vid_name))
    sys.stdout.flush()
    return file_list

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='extract frames from videos')
    parser.add_argument('src_dir', type=str, default='UCFCrime/MAR_DATA', nargs='?')
    parser.add_argument('out_dir', type=str, default='UCFCrime/frames', nargs='?')
    parser.add_argument('--ext', type=str, default='mp4', choices=['avi', 'mp4'], help='video file extensions')

    args = parser.parse_args()

    out_path = args.out_dir
    src_path = args.src_dir
    ext = args.ext

    if not os.path.isdir(out_path):
        print('creating folder: ' + out_path)
        os.makedirs(out_path)

    vid_list = glob.glob(src_path + '/*/*.' + ext)
    print('total number of videos found: ', len(vid_list))

    try:
        os.mkdir(out_path)
    except OSError:
        pass

    for vid in vid_list:
        dump_frames(vid, len(vid_list))
