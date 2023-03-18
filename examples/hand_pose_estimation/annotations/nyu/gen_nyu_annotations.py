import sys
import os
import scipy.io as sio
import numpy as np

# root dir of NYU dataset
dataset_dir = '/home/workspace/Datasets/NYU/'
save_dir = './'
data_type = ['train', 'test']
active_joint_idx = [0,3,6,9,12,15,18,21,24,25,27,30,31,32]
kinect_index = 0 # only use the frontal view according to the evaluation protocol

# generate file list and labels
for subset in data_type:
    # load joint annotations
    mat_file = os.path.join(dataset_dir, subset, 'joint_data.mat')
    joint_data = sio.loadmat(mat_file)
    joints_uvd = joint_data['joint_uvd']
    joints_uvd = np.squeeze(joints_uvd[kinect_index, :, active_joint_idx, :])
    joints_uvd = np.transpose(joints_uvd, (1, 0, 2)) # 14xNx3 -> Nx14x3
    # save joint annotations
    print '{} images for {}'.format(joints_uvd.shape[0], subset)
    joints_uvd = np.reshape(joints_uvd, (joints_uvd.shape[0], -1))
    label_file = os.path.join(save_dir, '{}_label.txt'.format(subset))
    np.savetxt(label_file, joints_uvd, delimiter=' ')
    with open(os.path.join(save_dir, '{}_image.txt'.format(subset)), 'w') as image_file:
        for image_idx in xrange(joints_uvd.shape[0]):
            filename = '{}/depth_{}_{}.png'.format(subset, kinect_index+1, str(image_idx+1).zfill(7))
            image_file.write(filename + "\n")
