import caffe
from caffe import layers as L, params as P
from caffe.coord_map import crop

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

    
def hand_baseline_wide(net_type):
    n = caffe.NetSpec()
    fx_ = 588.03
    fy_ = 587.07
    ux_ = 320
    uy_ = 240
    root_folder_ = '/home/workspace/Datasets/NYU/'
    point_num_ = 14
    # data layers
    if net_type == 'train':
        image_source_=root_folder_+'train_img.txt'
        pose_data_param_train = dict(image_source=image_source_,
                                     label_source=root_folder_+'train_label.txt',
                                     root_folder=root_folder_,
                                     batch_size=128, shuffle=True, new_height=96, new_width=96,
                                     point_num=point_num_, point_dim=3,
                                     cube_length=150, fx=fx_, fy=fy_, dataset=P.PoseData.NYU)
        n.data, n.pose = L.PoseData(name="data", include=dict(phase=0),
                     transform_param=dict(is_trans=True, trans_dx=10, trans_dy=10, is_rotate=True, rotate_deg=15, is_zoom=True, zoom_scale=0.1),
                     pose_data_param=pose_data_param_train, ntop=2)
        first_layer = str(n.to_proto())
        
        pose_data_param_test = dict(image_source=root_folder_+'test_img.txt',
                                 label_source=root_folder_+'test_label.txt',
                                 root_folder=root_folder_,
                                 batch_size=128, shuffle=False, new_height=96, new_width=96,
                                 point_num=point_num_, point_dim=3, output_center=True,
                                 cube_length=150, fx=fx_, fy=fy_, dataset=P.PoseData.NYU)
        n.data, n.label = L.PoseData(name="data", include=dict(phase=1),
                     transform_param=dict(is_trans=False, is_rotate=False, is_zoom=False),
                     pose_data_param=pose_data_param_test, ntop=2)
        n.pose, n.center = L.Slice(n.label, slice_param=dict(slice_dim=1, slice_point=point_num_*3), include=dict(phase=1), ntop=2)
    elif net_type == 'test-test':
        pose_data_param_test = dict(image_source=root_folder_+'test_img.txt',
                                 label_source=root_folder_+'test_label.txt',
                                 root_folder=root_folder_,
                                 batch_size=128, shuffle=False, new_height=96, new_width=96,
                                 point_num=point_num_, point_dim=3, output_center=True,
                                 cube_length=150, fx=fx_, fy=fy_, dataset=P.PoseData.NYU)
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
    n.fc1_0, n.relu6_0, n.drop1_0 = fc_relu_dropout(n.relu5, 2048, 0.5)
    n.fc2_0, n.relu7_0, n.drop2_0 = fc_relu_dropout(n.drop1_0, 2048, 0.5)
    n.fc3_0 = fc(n.drop2_0, point_num_*3)


    # loss
    if net_type == 'train':
        n.loss = L.SmoothL1Loss(n.fc3_0, n.pose,
                smooth_l1_loss_param=dict(sigma=10),
                loss_weight=1)
        n.distance = L.PoseDistance(n.fc3_0, n.pose, n.center, loss_weight=0,
                                    pose_distance_param=dict(cube_length=150, fx=fx_, fy=fy_, ux=ux_, uy=uy_),
                                    include=dict(phase=1))
        return first_layer + str(n.to_proto())
    else:
        n.error, n.output = L.PoseDistance(n.fc3_0, n.pose, n.center,
                                    pose_distance_param=dict(cube_length=150, fx=fx_, fy=fy_, ux=ux_, uy=uy_, output_pose=True),
                                    include=dict(phase=1), ntop=2)
        return str(n.to_proto())

def make_solver():
    solver_content = 'net: \"model_nyu/handpose_baseline_nyu.prototxt\"\n\
test_iter: 64\n\
test_interval: 1000\n\
base_lr: 0.001\n\
lr_policy: \"step\"\n\
gamma: 0.1\n\
stepsize: 40000\n\
display: 100\n\
max_iter: 160000\n\
momentum: 0.9\n\
weight_decay: 0.0005\n\
snapshot: 160000\n\
snapshot_prefix: \"snapshot_nyu/handpose_baseline_nyu\"'
    return solver_content

def make_net():
    with open('model_nyu/handpose_baseline_nyu.prototxt', 'w') as f:
        f.write(hand_baseline_wide('train'))
    with open('model_nyu/test_handpose_baseline_nyu.prototxt', 'w') as f:
        f.write(hand_baseline_wide('test-test'))
    with open('model_nyu/solver_handpose_baseline_nyu.prototxt', 'w') as f:
        f.write(make_solver())

if __name__ == '__main__':
    make_net()
