import numpy as np
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'model_msra'))
sys.path.append(os.path.join(BASE_DIR, '../../../python'))
import caffe
print os.path.join(BASE_DIR, '../../../python')
from net_handpose_baseline_msra import make_net

def make_output_pose_command(output_pose_bin, model, weights, label_list, output_name, fx, fy, ux, uy, test_id):
    command = '{0} \
    --model={1} \
    --gpu=0 \
    --weights={2} \
    --label_list={3} \
    --output_name={4} \
    --fx={5} \
    --fy={6} \
    --ux={7} \
    --uy={8} \
    2>&1 | tee logs/test_handpose_baseline_msra_{9}.txt'.format(output_pose_bin, model, weights, label_list, output_name, fx, fy, ux, uy, test_id)
    return command

# make net
# make_net()

# init caffe
caffe.set_device(0)
caffe.set_mode_gpu()

# parameters
root_dir = '/home/workspace/Datasets/MSRA/cvpr15_MSRAHandGestureDB/'
output_pose_bin = '../../../build/tools/output_pose'
fx = 240.99
fy = 240.96
ux = 160
uy = 120

test_id = sys.argv[1]

print 'test_id: {}'.format(test_id)

# --------------------------------------------------------------------------
# test
# --------------------------------------------------------------------------
print 'start testing ...'
# prepare input files
model = 'model_msra/test_handpose_baseline_msra_{}.prototxt'.format(test_id)
weights = 'snapshot_msra/handpose_baseline_msra_{}_iter_80000.caffemodel'.format(test_id)
output_name = 'output/test_handpose_baseline_msra_{}.prototxt'.format(test_id)
label_list = root_dir + 'test_label_{}.txt'.format(test_id)
cmd = make_output_pose_command(output_pose_bin, model, weights, label_list, output_name, fx, fy, ux, uy, test_id)
print cmd
os.system(cmd)
print 'finish testing ...'
