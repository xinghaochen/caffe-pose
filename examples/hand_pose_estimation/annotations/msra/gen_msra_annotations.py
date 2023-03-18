import sys
import os
import scipy.io as sio
import numpy as np

# root dir of NYU dataset
dataset_dir = '/home/workspace/Datasets/MSRA/cvpr15_MSRAHandGestureDB'
dataset_dir = '/home/workspace/data/handpose/MSRA/cvpr15_MSRAHandGestureDB'

save_dir = './'
folder_list = ['1','2','3','4','5','6','7','8','9','I','IP','L','MP','RP','T','TIP','Y']

f = 241.42
x = 160
y = 120

# generate file list and labels
image_list_all = []
label_list_all = []
for test_id in xrange(9):
    image_list_id = []
    label_list_id = []
    for ges_id in xrange(len(folder_list)):
        folder_dir = 'P{}/{}'.format(test_id, folder_list[ges_id])
        print os.path.join(dataset_dir, folder_dir, 'joint.txt')
        with open(os.path.join(dataset_dir, folder_dir, 'joint.txt'), 'r') as joint_file:
            lines = joint_file.readlines()
            nline = int(lines[0]) # the first indicates the number of frames in this folder
            joints = [map(float, line.split()) for line in lines[1:]]
            label_list_id.append(joints)
            for image_idx in xrange(nline):
                filename = '{}/{}_depth.bin'.format(folder_dir, str(image_idx).zfill(6))
                image_list_id.append(filename)
    image_list_all.append(image_list_id)
    label_list_all.append(label_list_id)

# save files for every split
for test_id in xrange(9):
    # file list
    test_image = image_list_all[test_id]
    train_image = [image_list_all[idx] for idx in xrange(9) if idx != test_id]
    # flatten the list
    train_image = [item for sublist in train_image for item in sublist]
    print len(test_image), len(train_image)
 
    # annotations
    test_label = label_list_all[test_id]
    train_label = [label_list_all[idx] for idx in xrange(9) if idx != test_id]
    # flatten the list
    test_label = np.concatenate(test_label)
    train_label = [item for sublist in train_label for item in sublist]
    train_label = np.concatenate(train_label)
    print test_label.shape, train_label.shape
	# convert xyz to uvd
    test_label = np.reshape(test_label, [test_label.shape[0], 21, 3])
    test_label[:,:,1] = -test_label[:,:,1]
    test_label[:,:,2] = -test_label[:,:,2]
    test_label[:,:,0] = test_label[:,:,0] * f / test_label[:,:,2] + x
    test_label[:,:,1] = test_label[:,:,1] * f / test_label[:,:,2] + y
    test_label = np.reshape(test_label, [test_label.shape[0], -1])

    train_label = np.reshape(train_label, [train_label.shape[0], 21, 3])
    train_label[:,:,1] = -train_label[:,:,1]
    train_label[:,:,2] = -train_label[:,:,2]
    train_label[:,:,0] = train_label[:,:,0] * f / train_label[:,:,2] + x
    train_label[:,:,1] = train_label[:,:,1] * f / train_label[:,:,2] + y
    train_label = np.reshape(train_label, [train_label.shape[0], -1])

    # save
    label_file = os.path.join(save_dir, 'test_label_{}.txt'.format(test_id))
    np.savetxt(label_file, test_label, fmt='%.04f')

    label_file = os.path.join(save_dir, 'train_label_{}.txt'.format(test_id))
    np.savetxt(label_file, train_label, fmt='%.04f')

    image_file = os.path.join(save_dir, 'test_image_{}.txt'.format(test_id))
    with open(image_file, 'w') as fp:
        fp.write('\n'.join(test_image))
    image_file = os.path.join(save_dir, 'train_image_{}.txt'.format(test_id))
    with open(image_file, 'w') as fp:
        fp.write('\n'.join(train_image))

# save all list
image_list_all = [item for sublist in image_list_all for item in sublist]
label_list_all = [item for sublist in label_list_all for item in sublist]
label_list_all = np.concatenate(label_list_all)

label_file = os.path.join(save_dir, 'all_label.txt')
np.savetxt(label_file, label_list_all, fmt='%.04f')

image_file = os.path.join(save_dir, 'all_image.txt')
with open(image_file, 'w') as fp:
    fp.write('\n'.join(image_list_all))
print len(label_list_all), len(image_list_all)

