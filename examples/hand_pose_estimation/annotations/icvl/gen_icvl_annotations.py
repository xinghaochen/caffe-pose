import sys
import os

# root dir of ICVL dataset
dataset_dir = '/home/workspace/Datasets/ICVL/'
save_dir = './'

# generate training file list and labels
train_file = os.path.join(dataset_dir, 'train/labels.txt')
with open(train_file, "r") as in_file:
    with open(os.path.join(save_dir, 'train_image.txt'), 'w') as image_file:
        with open(os.path.join(save_dir, 'train_label.txt'), 'w') as label_file:
            for line in in_file:
                items = line.split()
                image_file.write(items[0] + "\n")
                label_file.write(" ".join(items[1:]) + "\n")

# generate testing file list and labels
# test_seq_1
train_file = os.path.join(dataset_dir, 'test/test_seq_1.txt')
with open(train_file, "r") as in_file:
    with open(os.path.join(save_dir, 'test_image.txt'), 'w') as image_file:
        with open(os.path.join(save_dir, 'test_label.txt'), 'w') as label_file:
            for line in in_file:
                items = line.split()
                image_file.write(items[0] + "\n")
                label_file.write(" ".join(items[1:]) + "\n")
# test_seq_2
train_file = os.path.join(dataset_dir, 'test/test_seq_2.txt')
with open(train_file, "r") as in_file:
    with open(os.path.join(save_dir, 'test_image.txt'), 'a') as image_file:
        with open(os.path.join(save_dir, 'test_label.txt'), 'a') as label_file:
            for line in in_file:
                items = line.split()
                image_file.write(items[0] + "\n")
                label_file.write(" ".join(items[1:]) + "\n")
