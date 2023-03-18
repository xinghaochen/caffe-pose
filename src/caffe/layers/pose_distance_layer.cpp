/*************************************************************************
    > File Name: pose_distance_layer.cpp
    > Author: Guo Hengkai, Xinghao Chen
 ************************************************************************/

#include <functional>
#include <utility>
#include <vector>

#include "caffe/layers/pose_distance_layer.hpp"
#include "caffe/util/math_functions.hpp"

#define SQR(x) ((x)*(x))

namespace caffe {

template <typename Dtype>
PoseDistanceLayer<Dtype>::PoseDistanceLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {
    fx_ = param.pose_distance_param().fx();
    fy_ = param.pose_distance_param().fy();
    ux_ = param.pose_distance_param().ux();
    uy_ = param.pose_distance_param().uy();

    cube_length_.clear();

    for (int i = 0; i < param.pose_distance_param().cube_length_size(); ++i)
    {
        cube_length_.push_back(param.pose_distance_param().cube_length(i));
    }
    if (cube_length_.size() == 1)
    {
        cube_length_.resize(3, cube_length_[0]);
    }
    CHECK_EQ(cube_length_.size(), 3);

    output_pose_ = param.pose_distance_param().output_pose();
    ids_.clear();
    if (param.pose_distance_param().permute_id_size())
    {
        int n = param.pose_distance_param().permute_id_size();
        ids_.resize(n);
        for (int i = 0; i < n; ++i)
        {
            ids_[param.pose_distance_param().permute_id(i)] = i;
        }
    }
    LOG(INFO) << "Size of ids: " << ids_.size();
}

template <typename Dtype>
void PoseDistanceLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[0]->shape(0), bottom[1]->shape(0))
      << "The data and label should have the same first dimension.";
  CHECK_EQ(bottom[0]->shape(1), bottom[1]->shape(1))
      << "The data and label should have the same second dimension.";
  CHECK_EQ(bottom[0]->shape(0), bottom[2]->shape(0))
      << "The data and center should have the same first dimension.";
  CHECK(bottom[2]->shape(1) == 3 || bottom[2]->shape(1) == bottom[1]->shape(1))
      << "The center should have 3 elements or the same as label.";
  has_weight_ = (bottom.size() == 4);

  vector<int> top_shape(0);
  top[0]->Reshape(top_shape);
  if (output_pose_) {
    top[1]->ReshapeLike(*bottom[0]);
  }
}

template <typename Dtype>
void PoseDistanceLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();
  Dtype loss = 0;
  const Dtype* predict_data = bottom[0]->cpu_data();
  const Dtype* label_data = bottom[1]->cpu_data();
  const Dtype* centers = bottom[2]->cpu_data();
  const Dtype* weights = has_weight_ ? bottom[3]->cpu_data() : NULL;
  const bool is_single = bottom[2]->shape(1) != bottom[1]->shape(1);
  const int num = bottom[0]->num();
  const int label_num = count / num;
  Dtype* pose = NULL;
  if (output_pose_) {
    pose = top[1]->mutable_cpu_data();
  }
  int cnt = 0;
  for (int i = 0, k = 0, c = 0; i < num; ++i, k += label_num) {
    for (int j = 0; j < label_num; j += 3) {
      Dtype c1 = centers[c];
      Dtype c2 = centers[c + 1];
      Dtype c3 = centers[c + 2];
      Dtype u1 = predict_data[k + j] * cube_length_[0] * fabs(fx_) / c3 + c1;
      Dtype v1 = predict_data[k + j + 1] * cube_length_[1] * fabs(fy_) / c3 + c2;
      Dtype d1 = predict_data[k + j + 2] * cube_length_[2] + c3;
      Dtype x1 = (u1 - ux_) * d1 / fx_;
      Dtype y1 = (v1 - uy_) * d1 / fy_;
      Dtype u2 = label_data[k + j] * cube_length_[0] * fabs(fx_) / c3 + c1;
      Dtype v2 = label_data[k + j + 1] * cube_length_[1] * fabs(fy_) / c3 + c2;
      Dtype d2 = label_data[k + j + 2] * cube_length_[2] + c3;
      Dtype x2 = (u2 - ux_) * d2 / fx_;
      Dtype y2 = (v2 - uy_) * d2 / fy_;
      // if weights are explictly provided, only calculate error for weights == 1
      if (!has_weight_ || fabs(*weights - 1) < 1e-3) {
          loss += sqrt(SQR(x1 - x2) + SQR(y1 - y2) + SQR(d1 - d2));
          ++cnt;
      }
      if (has_weight_) {
          ++weights;
          ++weights;
          ++weights;
      }
      if (!is_single) {
        c += 3;
      }
      if (pose) {
        int idx = ids_.empty() ? j : ids_[j / 3] * 3;
        pose[k + idx] = u1;
        pose[k + idx + 1] = v1;
        pose[k + idx + 2] = d1;
      }
    }
    if (is_single) {
      c += 3;
    }
  }
  top[0]->mutable_cpu_data()[0] = loss / cnt;
}

#ifdef CPU_ONLY
STUB_GPU(PoseDistanceLayer);
#endif

INSTANTIATE_CLASS(PoseDistanceLayer);
REGISTER_LAYER_CLASS(PoseDistance);

}  // namespace caffe
