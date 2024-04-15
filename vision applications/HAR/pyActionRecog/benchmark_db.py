import glob
import os
import random


def parse_directory(path): 
    """
    Parse directories holding extracted frames from standard benchmarks
    """
    print('parse frames under folder {}'.format(path))
    frame_folders = glob.glob(os.path.join(path, '*'))

    def count_files(directory):
        lst = os.listdir(directory)
        cnt_list = len(lst)
        return cnt_list

    # check RGB
    rgb_counts = {}
    dir_dict = {}
    for i,f in enumerate(frame_folders):
        cnt = count_files(f)
        #print(cnt)
        k = f.split('/')[-1]
        rgb_counts[k] = cnt
        dir_dict[k] = f
        #print(dir_dict[k])
        if i % 200 == 0:
            print('{} videos parsed'.format(i))

    print('frame folder analysis done')
    return dir_dict, rgb_counts


def build_split_list(split_tuple, frame_info, split_idx, shuffle=False):
    split = split_tuple[split_idx]

    def build_set_list(set_list):
        rgb_list = list()
        for item in set_list:
            frame_dir = frame_info[0][item[0]]
            rgb_cnt = frame_info[1][item[0]]
            #print(item[1])
            rgb_list.append('{} {} {}\n'.format(frame_dir, rgb_cnt, item[1]))
        if shuffle:
            random.shuffle(rgb_list)
        return rgb_list

    train_rgb_list = build_set_list(split[0])
    test_rgb_list = build_set_list(split[1])
    #print(train_rgb_list)
    return train_rgb_list, test_rgb_list


# Dataset specific split file parse
def parse_ucf_splits():
    data_path='/media/ihsan/HDDP12/Workspace/ClassIE803/HAR/UCFCrime'
    class_ind = [x.strip().split() for x in open(data_path+'/labels/ClassIDs.txt')]
    #print(class_ind)
    class_mapping = {x[1]:int(x[0])-1 for x in class_ind}
    #print(class_mapping)
    def line2rec(line):
        items = line.strip().split('/')
        #print(items[0])
        label = class_mapping[items[0]]
        #print(label)
        vid = items[1].split('.')[0]
        #print(vid)
        return vid, label

    splits = []
    #train_list = [line2rec(x) for x in open(data_path + '/labels/train_001.txt')]
    #test_list = [line2rec(x) for x in open(data_path + '/labels/test_001.txt')]
    #splits.append((train_list, test_list))
    
    for i in range(1, 4):
        train_list = [line2rec(x) for x in open(data_path + '/labels/train_{:03d}.txt'.format(i))]
        test_list = [line2rec(x) for x in open(data_path + '/labels/test_{:03d}.txt'.format(i))]
        splits.append((train_list, test_list))
    return splits
