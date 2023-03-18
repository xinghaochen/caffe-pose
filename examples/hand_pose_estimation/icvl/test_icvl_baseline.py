import numpy as np
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'model_icvl'))
sys.path.append(os.path.join(BASE_DIR, '../../../python'))
import caffe
print os.path.join(BASE_DIR, '../../../python')
from net_handpose_baseline_icvl import make_net

def make_output_pose_command(output_pose_bin, model, weights, label_list, output_name, fx, fy, ux, uy):
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
    2>&1 | tee test_handpose_baseline_icvl.txt'.format(output_pose_bin, model, weights, label_list, output_name, fx, fy, ux, uy)
    return command

# make net
# make_net()

# init caffe
caffe.set_device(0)
caffe.set_mode_gpu()

# parameters
root_dir = '/home/workspace/Datasets/ICVL/'
output_pose_bin = '../../../build/tools/output_pose'
fx = 240.99
fy = 240.96
ux = 160
uy = 120

# --------------------------------------------------------------------------
# test
# --------------------------------------------------------------------------
print 'start testing ...'
# prepare input files
model = 'model_icvl/test_handpose_baseline_icvl.prototxt'
weights = 'snapshot_icvl/handpose_baseline_icvl_iter_160000.caffemodel'
output_name = 'output/test_handpose_baseline_icvl.txt'
label_list = root_dir + 'test_label.txt'
cmd = make_output_pose_command(output_pose_bin, model, weights, label_list, output_name, fx, fy, ux, uy)
print cmd
os.system(cmd)
print 'finish testing ...'
