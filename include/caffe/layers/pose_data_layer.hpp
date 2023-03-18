/*************************************************************************
    > File Name: pose_data_layer.hpp
    > Author: Guo Hengkai, Xinghao Chen, Cairong Zhang
 ************************************************************************/

#ifndef CAFFE_POSE_DATA_LAYER_HPP_
#define CAFFE_POSE_DATA_LAYER_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class PoseDataLayer : public BasePrefetchingDataLayer<Dtype> {
 public:
  struct PoseData {
    std::string name;
    cv::Mat img;
    vector<vector<float> > pose;
    vector<float> center;
    vector<float> bbox;
    PoseData(const std::string& n, const cv::Mat& i,
            const vector<vector<float> >& p, const vector<float>& c):
        name(n), img(i), pose(p), center(c) {
        // empty function
    }
    PoseData(const std::string& n, const cv::Mat& i,
            const vector<vector<float> >& p, const vector<float>& c, const vector<float>& w):
        name(n), img(i), pose(p), center(c) {
        // empty function
    }
    PoseData(const std::string& n, const cv::Mat& i,
            const vector<vector<float> >& p, const vector<float>& c, const vector<float>& w,
             const vector<float>& b):
        name(n), img(i), pose(p), center(c), bbox(b) {
        // empty function
    }
  };

  explicit PoseDataLayer(const LayerParameter& param)
      : BasePrefetchingDataLayer<Dtype>(param) {}
  virtual ~PoseDataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "PoseData"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int MinTopBlobs() const { return 2; }
  virtual inline int MaxTopBlobs() const { return 3; }

 protected:
  shared_ptr<Caffe::RNG> prefetch_rng_;
  virtual void ShuffleImages();
  virtual void load_batch(Batch<Dtype>* batch);
  double RandU(double min_val, double max_val);

  vector<PoseData> lines_;
  vector<size_t> idx_;
  int lines_id_;

  Blob<Dtype> transformed_label_;
  vector<int> cube_length_;
  PoseDataParameter_Dataset dataset_;
};


}  // namespace caffe

#endif  // CAFFE_POSE_DATA_LAYER_HPP_
