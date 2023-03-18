/*************************************************************************
    > File Name: pose_distance_layer.hpp
    > Author: Guo Hengkai, Xinghao Chen
 ************************************************************************/

#ifndef CAFFE_POSE_DISTANCE_LAYER_HPP_
#define CAFFE_POSE_DISTANCE_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class PoseDistanceLayer : public Layer<Dtype> {
 public:
  explicit PoseDistanceLayer(const LayerParameter& param);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "PoseDistance"; }
  virtual inline int MinBottomBlobs() const { return 3; }
  virtual inline int MaxBottomBlobs() const { return 4; }

  virtual inline int ExactNumTopBlobs() const { return 1 + output_pose_; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);


  /// @brief Not implemented.
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    for (int i = 0; i < propagate_down.size(); ++i) {
      if (propagate_down[i]) { NOT_IMPLEMENTED; }
    }
  }

  double fx_;
  double fy_;
  double ux_;
  double uy_;
  bool output_pose_;
  vector<int> ids_;
  vector<int> cube_length_;
  bool has_weight_;
};

}  // namespace caffe

#endif  // CAFFE_POSE_DISTANCE_LAYER_HPP_
