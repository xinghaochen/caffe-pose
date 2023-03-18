/*************************************************************************
    > File Name: pose_data_layer.cpp
    > Author: Guo Hengkai, Xinghao Chen, Cairong Zhang
 ************************************************************************/

#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>

#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/layers/pose_data_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/hdf5.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"


namespace caffe {

template <typename Dtype>
PoseDataLayer<Dtype>::~PoseDataLayer<Dtype>() {
  this->StopInternalThread();
}

template <typename Dtype>
void PoseDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // flags for debug
  bool debug_show = false;
  bool debug_save_crop = false;

  PoseDataParameter pose_data_param = this->layer_param_.pose_data_param();
  const int new_height = pose_data_param.new_height();
  const int new_width  = pose_data_param.new_width();
  string root_folder = pose_data_param.root_folder();

  CHECK((new_height == 0 && new_width == 0) ||
    (new_height > 0 && new_width > 0)) << "Current implementation requires "
    "new_height and new_width to be set at the same time.";

  const int target_joint = pose_data_param.target_joint();
  const int ori_point_num = pose_data_param.point_num();
  // Only predict a specific joint is target_joint is provided
  // Otherwise, predict all joints
  const int point_num = target_joint >= 0 ? 1 : ori_point_num;
  const int point_dim = pose_data_param.point_dim();
  const int label_num = point_num * point_dim;

  // cube size for cropping 
  cube_length_.clear();
  for (int i = 0; i < pose_data_param.cube_length_size(); ++i)
  {
    cube_length_.push_back(pose_data_param.cube_length(i));
  }
  if (cube_length_.size() == 1)
  {
    cube_length_.resize(3, cube_length_[0]);
  }
  CHECK_EQ(cube_length_.size(), 3);
  
  // intrinsic parameters of camera
  float fx = pose_data_param.fx();
  float fy = pose_data_param.fy();

  // Read the file with filenames and labels
  const string& image_source = pose_data_param.image_source();
  const string& label_source = pose_data_param.label_source();
  // A file that contains the centers of all images
  const string& center_source = pose_data_param.center_source();
  // A file that contains  bounding boxes of hands
  const string& bbox_source = pose_data_param.bbox_source();
  const bool use_center_from_file = !center_source.empty();
  const bool use_bbox = !bbox_source.empty();
  const bool pre_crop = pose_data_param.pre_crop();
  const bool lazy_load = pose_data_param.lazy_load();
  const int limit_num = pose_data_param.limit_num();
  bool full_input = pose_data_param.full_input();
  dataset_ = pose_data_param.dataset();

  vector<int> ids;
  if (pose_data_param.permute_id_size()) {
    CHECK_EQ(pose_data_param.permute_id_size(), ori_point_num);
    for (int i = 0; i < ori_point_num; ++i) {
      ids.push_back(pose_data_param.permute_id(i));
    }
  } else {
    for (int i = 0; i < ori_point_num; ++i) {
      ids.push_back(i);
    }
  }
  /*
  CHECK((pre_crop && !use_center_from_file) || !pre_crop)
      << "when using centers from file, pre_crop must be false";
  */
  CHECK((pre_crop && !lazy_load) || !pre_crop)
      << "when using lazy load mode, pre_crop must be false";
  vector<vector<float> > labels;
  std::ifstream infile_center;
  if (use_center_from_file) {
    infile_center.open(center_source.c_str());
  }
  std::ifstream infile_bbox;
  if (use_bbox) {
    infile_bbox.open(bbox_source.c_str());
  }
  
  // Hand pose dataset: ICVL, MSRA, NYU, Hands17
  int total_images = pose_data_param.total_images();
  LOG(INFO) << "Opening file " << image_source;
  std::ifstream infile_image(image_source.c_str());
  LOG(INFO) << "Opening file " << label_source;
  std::ifstream infile_label(label_source.c_str());
  string line;
  float label;
  vector<vector<float> > centers;
  // loop for all files in dataset
  while (std::getline(infile_image, line)) {
    if (line.empty() || line[0] == '#') {
      LOG(INFO) << "Skip " << line;
      if (total_images > 0 && idx_.size() < total_images) {
        --total_images;
      }
      continue;
    }
    labels.resize(ori_point_num);
    // load a pose annotation for a frame
    for (int i = 0; i < ori_point_num; ++i) {
      vector<float> tmp;
      for (int j = 0; j < point_dim; ++j) {
        infile_label >> label;
        tmp.push_back(label);
      }
      labels[ids[i]] = tmp;
    }

    // load a depth image
    vector<float> center;
    vector<float> bbox;
    cv::Mat cv_img;
    string name = root_folder + line;
    // early_load mode: load the data and store in memory to speed up training
    if (!lazy_load) {
      cv_img = ReadDepthImageToCVMat(name, dataset_); 
      if (!cv_img.data) {
        if (total_images > 0 && idx_.size() < total_images) {
          --total_images;
        }
        continue;
      }
    }
    // if bounding boxes are given
    if (use_bbox) {
      bbox.resize(4);
      for (int i = 0; i < 4; ++i) {
        infile_bbox >> bbox[i];
      }
    }
    // load centers from file
    if (use_center_from_file) {
      center.resize(point_dim);
      for (int i = 0; i < point_dim; ++i) {
        infile_center >> label;
        center[i] = label;
      }
    }
    // calculate center from image 
    else {
      size_t pos = line.find_first_of('/');
      cv::Mat img = (dataset_ == PoseDataParameter_Dataset_NYU) ?
        ReadDepthImageToCVMat(root_folder + line.substr(0, pos + 1) + "synth" + line.substr(pos + 1),
                    dataset_)
        : (lazy_load ? ReadDepthImageToCVMat(name, dataset_) : cv_img);
      if (dataset_ == PoseDataParameter_Dataset_ICVL) {
        GetCenter(img, center, 0, 500);
      }
      else if (dataset_ == PoseDataParameter_Dataset_HANDS17)
        GetCenterByBboxFast(img, bbox, center, 1, 800);
      else if (dataset_ == PoseDataParameter_Dataset_MSRA)
        GetCenter(img, center, 10, 1000);
      else if (dataset_ == PoseDataParameter_Dataset_NYU)
        GetCenter(img, center, 500, 1300);
      if (center.empty())
      {
        LOG(INFO) << "Skip " << line << " due to center";
        if (total_images > 0 && idx_.size() < total_images) {
          --total_images;
        }
        continue;
      }
    }

    // Debug only: visualize the image, pose annotation and center
    if (debug_show && !lazy_load) {
      std::cout << center[0] << " " << center[1] << " " << center[2] << std::endl;
      cv::Mat show(cv_img.clone());
      show /= 5000;
      cv::cvtColor(show, show, CV_GRAY2BGR);
      for (int i = 0; i < ori_point_num; ++i) {
        cv::circle(show, cv::Point2f(labels[i][0], labels[i][1]), 3,
                cv::Scalar(0, 0, 255), -1);
      }
      cv::circle(show, cv::Point2f(center[0], center[1]), 3,
                  cv::Scalar(0, 255, 0), -1);
      cv::imshow("", show);
      char ch = cv::waitKey();
      if (ch == 'q') {
        exit(0);
      }
    }
    
    // crop the depth image to get hand patch
    if (!lazy_load && pre_crop) {
      if (full_input) {
        // TODO
      } else {
        // Hands17 dataset provides bounding boxes for frame-based task
        if (use_bbox)
          cv_img = CropImageInBbox(cv_img, bbox, labels, center, cube_length_,
              fx, fy, new_height, new_width);
        else
          cv_img = CropImage(cv_img, labels, center, cube_length_,
              fx, fy, new_height, new_width);
      }
    }

    // Debug only: save the cropped image
    if (debug_save_crop) {
      std::stringstream ss;
      ss << lines_.size();
      cv::Mat save_img(cv_img.clone());
      save_img /= 2;
      save_img += 0.5;
      save_img *= 255;
      save_img.convertTo(save_img, CV_8U);
      cv::imwrite("cropped/" + ss.str() + ".jpg", save_img);
      if (debug_show)
          cv::imshow("crop", save_img);
    }

    // if target_joint is explictly given
    // only predict this joint and discard others
    if (target_joint >= 0) {
      labels[0] = labels[target_joint];
      labels.erase(labels.begin() + 1, labels.end());
    }

    // if total_images is explictly given and currently loaded iamges are more than total_images
    // reuse images for to save memory
    if (total_images && idx_.size() >= total_images){
      if (idx_.size() == total_images) {
        LOG(INFO) << "Loading the same images again...";
      }
      lines_.push_back(PoseData(name, lines_[idx_.size() % total_images].img, labels, center, bbox));
    }
    // otherwise just save the images into memory
    else {
      lines_.push_back(PoseData(name, cv_img.clone(), labels, center, bbox));
    }
    // logging
    idx_.push_back(idx_.size());
    LOG_EVERY_N(INFO, 2000) << "Loading " << idx_.size() << " image...";
    // if limit_num is explictly given, only load no more than limit_num images
    // just break the loop and stop loading images 
    if (limit_num > 0 && lines_.size() == limit_num) {
      break;
    }
  }
  // CHECK(false);

  CHECK(!lines_.empty()) << "File is empty";

  if (pose_data_param.shuffle()) {
    // randomly shuffle data
    LOG(INFO) << "Shuffling data";
    const unsigned int prefetch_rng_seed = caffe_rng_rand();
    prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
    ShuffleImages();
  }
  LOG(INFO) << "A total of " << lines_.size() << " images.";

  lines_id_ = 0;
  // Check if we would need to randomly skip a few data points
  if (pose_data_param.rand_skip()) {
    unsigned int skip = caffe_rng_rand() %
        pose_data_param.rand_skip();
    LOG(INFO) << "Skipping first " << skip << " data points.";
    CHECK_GT(lines_.size(), skip) << "Not enough points to skip";
    lines_id_ = skip;
  }
  // Read an image, and use it to initialize the top blob.
  cv::Mat cv_img = lazy_load ? ReadDepthImageToCVMat(lines_[idx_[lines_id_]].name, dataset_)
                              : lines_[idx_[lines_id_]].img;
  CHECK(cv_img.data) << "Could not load " << lines_[idx_[lines_id_]].name;

  // get cropped image
  vector<float> center = lines_[idx_[lines_id_]].center;
  vector<float> bbox = lines_[idx_[lines_id_]].bbox;
  static vector<vector<float> > dummy_points;
  if (!pre_crop && !full_input) {
    if (use_bbox)
        cv_img = CropImageInBbox(cv_img, bbox, dummy_points, center, cube_length_,
            fx, fy, new_height, new_width);
    else
        cv_img = CropImage(cv_img, dummy_points, center, cube_length_,
            fx, fy, new_height, new_width);
  }
  
  // Use data_transformer to infer the expected blob shape from a cv_image.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(cv_img);
  this->transformed_data_.Reshape(top_shape);
  // Reshape prefetch_data and top[0] according to the batch_size.
  const int batch_size = pose_data_param.batch_size();
  CHECK_GT(batch_size, 0) << "Positive batch size required";
  top_shape[0] = batch_size;
  top_shape[1] = 1;
  for (int i = 0; i < this->prefetch_.size(); ++i) {
    this->prefetch_[i]->data_.Reshape(top_shape);
  }
  top[0]->Reshape(top_shape);

  LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();
  // label
  vector<int> label_shape;
  label_shape.push_back(batch_size);
  // concatenate hand pose together with center
  label_shape.push_back(label_num + pose_data_param.output_center() * center.size());
  top[1]->Reshape(label_shape);
  for (int i = 0; i < this->prefetch_.size(); ++i) {
    this->prefetch_[i]->label_.Reshape(label_shape);
  }

  if (target_joint >= 0) {
    label_shape[1] = 3;
  } else {
    label_shape[1] = label_num;
  }
  this->transformed_label_.Reshape(label_shape);
}

template <typename Dtype>
void PoseDataLayer<Dtype>::ShuffleImages() {
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  shuffle(idx_.begin(), idx_.end(), prefetch_rng);
}

// This function is called on prefetch thread
template <typename Dtype>
void PoseDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK(batch->data_.count());
  CHECK(this->transformed_data_.count());
  PoseDataParameter pose_data_param = this->layer_param_.pose_data_param();
  const int batch_size = pose_data_param.batch_size();
  const int target_joint = pose_data_param.target_joint();
  const int point_num = target_joint >= 0 ? 1 : pose_data_param.point_num();
  const int point_dim = pose_data_param.point_dim();
  bool output_center = pose_data_param.output_center();
  string root_folder = pose_data_param.root_folder();
  //const bool use_center_from_file = !pose_data_param.center_source().empty();
  const float fx = pose_data_param.fx();
  const float fy = pose_data_param.fy();
  const int new_height = pose_data_param.new_height();
  const int new_width  = pose_data_param.new_width();
  bool full_input = pose_data_param.full_input();
  const bool pre_crop = pose_data_param.pre_crop();
  const bool lazy_load = pose_data_param.lazy_load();
  const bool label_noise = pose_data_param.label_noise();
  const float label_std = pose_data_param.label_std();
  const string& bbox_source = pose_data_param.bbox_source();
  const bool use_bbox = !bbox_source.empty();

  static vector<vector<float> > dummy_points;

  // Reshape according to the first image of each batch
  // on single input batches allows for inputs of varying dimension.
  cv::Mat cv_img = lazy_load ? ReadDepthImageToCVMat(lines_[idx_[lines_id_]].name, dataset_)
      : lines_[idx_[lines_id_]].img;
  vector<float> center = lines_[idx_[lines_id_]].center;
  vector<float> bbox = lines_[idx_[lines_id_]].bbox;
  if (!pre_crop && !full_input) {
    if (use_bbox)
      cv_img = CropImageInBbox(cv_img, bbox, dummy_points, center, cube_length_,
            fx, fy, new_height, new_width);
    else
      cv_img = CropImage(cv_img, dummy_points, center, cube_length_,
            fx, fy, new_height, new_width);
  }
  const int label_num = point_num * point_dim + output_center * center.size();

  // Use data_transformer to infer the expected blob shape from a cv_img.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(cv_img);
  this->transformed_data_.Reshape(top_shape);
  // Reshape batch according to the batch_size.
  top_shape[0] = batch_size;
  top_shape[1] = 1;
  batch->data_.Reshape(top_shape);

  vector<int> label_shape;
  label_shape.push_back(batch_size);
  label_shape.push_back(label_num);
  batch->label_.Reshape(label_shape);
  if (target_joint >= 0) {
    label_shape[1] = 3;
  }
  this->transformed_label_.Reshape(label_shape);

  Dtype* prefetch_data = batch->data_.mutable_cpu_data();
  Dtype* prefetch_label = batch->label_.mutable_cpu_data();

  // datum scales
  const int lines_size = lines_.size();
  for (int item_id = 0; item_id < batch_size; ++item_id) {
    // get a blob
    timer.Start();
    CHECK_GT(lines_size, lines_id_);
    cv::Mat cv_img = lazy_load ? ReadDepthImageToCVMat(lines_[idx_[lines_id_]].name, dataset_)
        : lines_[idx_[lines_id_]].img;
    CHECK(cv_img.data) << "Could not load " << lines_[idx_[lines_id_]].name;
    vector<float> center = lines_[idx_[lines_id_]].center;
    vector<float> bbox = lines_[idx_[lines_id_]].bbox;
    read_time += timer.MicroSeconds();
    timer.Start();
    // Apply transformations (mirror, crop...) to the image
    int offset_label = batch->label_.offset(item_id);

    int offset_image = batch->data_.offset(item_id);
    this->transformed_data_.set_cpu_data(prefetch_data + offset_image);
    this->transformed_label_.set_cpu_data(prefetch_label + offset_label);
    vector<vector<float> > pose = lines_[idx_[lines_id_]].pose;
    vector<float> c = center;

    if (!pre_crop && !full_input) {
        if (dataset_ == PoseDataParameter_Dataset_HANDS17)
            cv_img = CropImageInBbox(cv_img, bbox, pose, c, cube_length_,
                fx, fy, new_height, new_width);
        else
            cv_img = CropImage(cv_img, pose, c, cube_length_,
                fx, fy, new_height, new_width);
    }
    if (label_noise) {
        for (size_t j = 0; j < pose.size(); ++j)
            for (size_t k = 0; k < pose[j].size(); ++k) {
                pose[j][k] += RandU(-label_std, label_std);
            }
    }
    this->data_transformer_->Transform(cv_img, pose,
            &(this->transformed_data_), &(this->transformed_label_));

    int offset_cur = offset_label + point_num * point_dim;
    if (output_center) {
      for (int i = 0; i < center.size(); ++i, ++offset_cur) {
        prefetch_label[offset_cur]
            = lines_[idx_[lines_id_]].center[i];
      }
    }
    trans_time += timer.MicroSeconds();

    // go to the next iter
    lines_id_++;
    if (lines_id_ >= lines_size) {
      // We have reached the end. Restart from the first.
      DLOG(INFO) << "Restarting data prefetching from start.";
      lines_id_ = 0;
      if (pose_data_param.shuffle()) {
        ShuffleImages();
      }
    }
  }
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

template <typename Dtype>
double PoseDataLayer<Dtype>::RandU(double min_val, double max_val) {
  caffe::rng_t* rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  double val = (double)((*rng)()) / rng->max();
  return min_val + (max_val - min_val) * val;
}

INSTANTIATE_CLASS(PoseDataLayer);
REGISTER_LAYER_CLASS(PoseData);

}  // namespace caffe
#endif  // USE_OPENCV