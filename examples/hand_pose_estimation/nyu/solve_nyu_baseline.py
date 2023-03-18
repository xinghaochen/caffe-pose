import numpy as np
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'model_nyu'))
sys.path.append(os.path.join(BASE_DIR, '../../../python'))
import caffe
from net_handpose_baseline_nyu import make_net

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
    2>&1 | tee logs/test_handpose_baseline_nyu.txt'.format(output_pose_bin, model, weights, label_list, output_name, fx, fy, ux, uy)
    return command

# make net
make_net()

# init caffe
caffe.set_device(0)
caffe.set_mode_gpu()

# parameters
root_dir = '/home/workspace/Datasets/NYU/'
output_pose_bin = '../../../build/tools/output_pose'
fx=588.03
fy=587.07
ux=320
uy=240

# --------------------------------------------------------------------------
# train
# --------------------------------------------------------------------------
print 'start training ...'
# solve

solver_name = 'model_nyu/solver_handpose_baseline_nyu.prototxt'
solver = caffe.SGDSolver(solver_name)
solver.solve()
print 'finish solving ...'

# --------------------------------------------------------------------------
# test
# --------------------------------------------------------------------------
print 'start testing ...'
# prepare input files
model = 'model_nyu/test_handpose_baseline_nyu.prototxt'
weights = 'snapshot_nyu/handpose_baseline_nyu_iter_160000.caffemodel'
output_name = 'output/test_handpose_baseline_nyu.txt'
label_list = root_dir + 'test_label.txt'
cmd = make_output_pose_command(output_pose_bin, model, weights, label_list, output_name, fx, fy, ux, uy)
os.system(cmd)
print 'finish testing ...'
