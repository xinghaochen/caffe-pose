import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../../../python'))
import caffe
from caffe import layers as L, params as P

def conv_relu(bottom, nout, ks=3, stride=1, pad=1):
    conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
        num_output=nout, pad=pad, weight_filler=dict(type='msra'), bias_filler=dict(type='constant'),
        param=[dict(lr_mult=1), dict(lr_mult=2)])
    return conv, L.ReLU(conv, in_place=True)

def conv(bottom, nout, ks=3, stride=1, pad=1):
    conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
        num_output=nout, pad=pad, weight_filler=dict(type='msra'), bias_filler=dict(type='constant'),
        param=[dict(lr_mult=1), dict(lr_mult=2)])
    return conv

def max_pool(bottom, ks=2, stride=2):
    return L.Pooling(bottom, pool=P.Pooling.MAX, kernel_size=ks, stride=stride)

def fc(bottom, nout):
    fc = L.InnerProduct(bottom, num_output=nout,
        param=[dict(lr_mult=1), dict(lr_mult=2)],
        weight_filler=dict(type='gaussian', std=0.001),
        bias_filler=dict(type='constant'))
    return fc

def fc_relu_dropout(bottom, nout, dropout):
    fc = L.InnerProduct(bottom, num_output=nout,
        param=[dict(lr_mult=1), dict(lr_mult=2)],
        weight_filler=dict(type='gaussian', std=0.001),
        bias_filler=dict(type='constant'))
    return fc, L.ReLU(fc, in_place=True), L.Dropout(fc, dropout_ratio=dropout, in_place=True)

    
def hand_baseline_wide(net_type, test_id):
    n = caffe.NetSpec()
    fx_ = 241.42
    fy_ = 241.42
    ux_ = 160
    uy_ = 120
    root_folder_ = '/home/workspace/Datasets/MSRA/cvpr15_MSRAHandGestureDB/'
    point_num_ = 21
    # data layers
    if net_type == 'train':
        pose_data_param_train = dict(image_source=root_folder_+'train_image_{}.txt'.format(test_id),
                                     label_source=root_folder_+'train_label_{}.txt'.format(test_id),
                                     root_folder=root_folder_,
                                     batch_size=128, shuffle=True, new_height=96, new_width=96,
                                     point_num=point_num_, point_dim=3, dataset=P.PoseData.MSRA,
                                     cube_length=150, fx=fx_, fy=fy_)
        n.data, n.pose = L.PoseData(name="data", include=dict(phase=0),
                     transform_param=dict(is_trans=True, trans_dx=10, trans_dy=10, is_rotate=True, rotate_deg=180, is_zoom=True, zoom_scale=0.1, mirror=False),
                     pose_data_param=pose_data_param_train, ntop=2)
        first_layer = str(n.to_proto())

        pose_data_param_test = dict(image_source=root_folder_+'test_image_{}.txt'.format(test_id),
                                 label_source=root_folder_+'test_label_{}.txt'.format(test_id),
                                 root_folder=root_folder_,
                                 batch_size=128, shuffle=False, new_height=96, new_width=96,
                                 point_num=point_num_, point_dim=3, output_center=True,
                                 dataset=P.PoseData.MSRA, cube_length=150, fx=fx_, fy=fy_)
        n.data, n.label = L.PoseData(name="data", include=dict(phase=1),
                     transform_param=dict(is_trans=False, is_rotate=False, is_zoom=False),
                     pose_data_param=pose_data_param_test, ntop=2)
        n.pose, n.center = L.Slice(n.label, slice_param=dict(slice_dim=1, slice_point=point_num_*3), include=dict(phase=1), ntop=2)
    elif net_type == 'test-test':
        pose_data_param_test = dict(image_source=root_folder_+'test_image_{}.txt'.format(test_id),
                                 label_source=root_folder_+'test_label_{}.txt'.format(test_id),
                                 root_folder=root_folder_,
                                 batch_size=128, shuffle=False, new_height=96, new_width=96,
                                 point_num=point_num_, point_dim=3, output_center=True,
                                 dataset=P.PoseData.MSRA, cube_length=150, fx=fx_, fy=fy_)
        n.data, n.label = L.PoseData(name="data", include=dict(phase=1),
                     transform_param=dict(is_trans=False, is_rotate=False, is_zoom=False),
                     pose_data_param=pose_data_param_test, ntop=2)
        n.pose, n.center = L.Slice(n.label, slice_param=dict(slice_dim=1, slice_point=point_num_*3), include=dict(phase=1), ntop=2)
    # the base net
    n.conv0, n.relu0 = conv_relu(n.data, 16)
    n.conv1 = conv(n.relu0, 16)
    n.pool1 = max_pool(n.conv1)
    n.relu1 = L.ReLU(n.pool1, in_place=True)

    n.conv2_0, n.relu2_0 = conv_relu(n.pool1, 32, ks=1, pad=0)
    n.conv2, n.relu2 = conv_relu(n.relu2_0, 32)
    n.conv3 = conv(n.relu2, 32)
    n.res1 = L.Eltwise(n.conv2_0, n.conv3)
    n.pool2 = max_pool(n.res1)
    n.relu3 = L.ReLU(n.pool2, in_place=True)

    n.conv3_0, n.relu3_0 = conv_relu(n.relu3, 64, ks=1, pad=0)
    n.conv4, n.relu4 = conv_relu(n.relu3_0, 64)
    n.conv5 = conv(n.relu4, 64)
    n.res2 = L.Eltwise(n.conv3_0, n.conv5)
    n.pool3 = max_pool(n.res2)
    n.relu5 = L.ReLU(n.pool3, in_place=True)

    # fc
    n.fc1, n.relu6, n.drop1 = fc_relu_dropout(n.relu5, 2048, 0.5)
    n.fc2, n.relu7, n.drop2 = fc_relu_dropout(n.drop1, 2048, 0.5)
    n.fc3 = fc(n.drop2, point_num_*3)

    # loss
    if net_type == 'train':
        n.loss = L.SmoothL1Loss(n.fc3, n.pose,
                smooth_l1_loss_param=dict(sigma=10),
                loss_weight=1)
        n.distance = L.PoseDistance(n.fc3, n.pose, n.center, loss_weight=0,
                                    pose_distance_param=dict(cube_length=150, fx=fx_, fy=fy_, ux=ux_, uy=uy_),
                                    include=dict(phase=1))
        return first_layer + str(n.to_proto())
    else:
        n.error, n.output = L.PoseDistance(n.fc3, n.pose, n.center,
                                    pose_distance_param=dict(cube_length=150, fx=fx_, fy=fy_, ux=ux_, uy=uy_, output_pose=True),
                                    include=dict(phase=1), ntop=2)
        return str(n.to_proto())

def make_solver(test_id):
    solver_content = 'net: \"model_msra/handpose_baseline_msra_{0}.prototxt\"\n\
test_iter: 67\n\
test_interval: 1000\n\
base_lr: 0.01\n\
lr_policy: \"step\"\n\
gamma: 0.1\n\
stepsize: 20000\n\
display: 100\n\
max_iter: 80000\n\
momentum: 0.9\n\
weight_decay: 0.0005\n\
snapshot: 40000\n\
snapshot_prefix: \"snapshot_msra/handpose_baseline_msra_{0}\"'.format(test_id)
    return solver_content

def make_net():
    for test_id in xrange(9):
        with open('model_msra/handpose_baseline_msra_{}.prototxt'.format(test_id), 'w') as f:
            f.write(hand_baseline_wide('train', test_id))
        with open('model_msra/test_handpose_baseline_msra_{}.prototxt'.format(test_id), 'w') as f:
            f.write(hand_baseline_wide('test-test', test_id))
        with open('model_msra/solver_handpose_baseline_msra_{}.prototxt'.format(test_id), 'w') as f:
            f.write(make_solver(test_id))

if __name__ == '__main__':
    make_net()
