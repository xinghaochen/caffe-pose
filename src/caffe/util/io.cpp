
#include <fcntl.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>
#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>
#endif  // USE_OPENCV
#include <stdint.h>

#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"

const int kProtoReadBytesLimit = INT_MAX;  // Max size of 2 GB minus 1 byte.

namespace caffe {

using google::protobuf::io::FileInputStream;
using google::protobuf::io::FileOutputStream;
using google::protobuf::io::ZeroCopyInputStream;
using google::protobuf::io::CodedInputStream;
using google::protobuf::io::ZeroCopyOutputStream;
using google::protobuf::io::CodedOutputStream;
using google::protobuf::Message;

bool ReadProtoFromTextFile(const char* filename, Message* proto) {
  int fd = open(filename, O_RDONLY);
  CHECK_NE(fd, -1) << "File not found: " << filename;
  FileInputStream* input = new FileInputStream(fd);
  bool success = google::protobuf::TextFormat::Parse(input, proto);
  delete input;
  close(fd);
  return success;
}

void WriteProtoToTextFile(const Message& proto, const char* filename) {
  int fd = open(filename, O_WRONLY | O_CREAT | O_TRUNC, 0644);
  FileOutputStream* output = new FileOutputStream(fd);
  CHECK(google::protobuf::TextFormat::Print(proto, output));
  delete output;
  close(fd);
}

bool ReadProtoFromBinaryFile(const char* filename, Message* proto) {
  int fd = open(filename, O_RDONLY);
  CHECK_NE(fd, -1) << "File not found: " << filename;
  ZeroCopyInputStream* raw_input = new FileInputStream(fd);
  CodedInputStream* coded_input = new CodedInputStream(raw_input);
  coded_input->SetTotalBytesLimit(kProtoReadBytesLimit, 536870912);

  bool success = proto->ParseFromCodedStream(coded_input);

  delete coded_input;
  delete raw_input;
  close(fd);
  return success;
}

void WriteProtoToBinaryFile(const Message& proto, const char* filename) {
  fstream output(filename, ios::out | ios::trunc | ios::binary);
  CHECK(proto.SerializeToOstream(&output));
}

#ifdef USE_OPENCV
cv::Mat ReadImageToCVMat(const string& filename,
    const int height, const int width, const bool is_color) {
  cv::Mat cv_img;
  int cv_read_flag = (is_color ? CV_LOAD_IMAGE_COLOR :
    CV_LOAD_IMAGE_GRAYSCALE);
  cv::Mat cv_img_origin = cv::imread(filename, cv_read_flag);
  if (!cv_img_origin.data) {
    LOG(ERROR) << "Could not open or find file " << filename;
    return cv_img_origin;
  }
  if (height > 0 && width > 0) {
    cv::resize(cv_img_origin, cv_img, cv::Size(width, height));
  } else {
    cv_img = cv_img_origin;
  }
  return cv_img;
}

cv::Mat ReadImageToCVMat(const string& filename,
    const int height, const int width) {
  return ReadImageToCVMat(filename, height, width, true);
}

cv::Mat ReadImageToCVMat(const string& filename,
    const bool is_color) {
  return ReadImageToCVMat(filename, 0, 0, is_color);
}

cv::Mat ReadImageToCVMat(const string& filename) {
  return ReadImageToCVMat(filename, 0, 0, true);
}

// Do the file extension and encoding match?
static bool matchExt(const std::string & fn,
                     std::string en) {
  size_t p = fn.rfind('.');
  std::string ext = p != fn.npos ? fn.substr(p) : fn;
  std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
  std::transform(en.begin(), en.end(), en.begin(), ::tolower);
  if ( ext == en )
    return true;
  if ( en == "jpg" && ext == "jpeg" )
    return true;
  return false;
}

bool ReadImageToDatum(const string& filename, const int label,
    const int height, const int width, const bool is_color,
    const std::string & encoding, Datum* datum) {
  cv::Mat cv_img = ReadImageToCVMat(filename, height, width, is_color);
  if (cv_img.data) {
    if (encoding.size()) {
      if ( (cv_img.channels() == 3) == is_color && !height && !width &&
          matchExt(filename, encoding) )
        return ReadFileToDatum(filename, label, datum);
      std::vector<uchar> buf;
      cv::imencode("."+encoding, cv_img, buf);
      datum->set_data(std::string(reinterpret_cast<char*>(&buf[0]),
                      buf.size()));
      datum->set_label(label);
      datum->set_encoded(true);
      return true;
    }
    CVMatToDatum(cv_img, datum);
    datum->set_label(label);
    return true;
  } else {
    return false;
  }
}

// --------------------------------------------------------------------
// functions below are added to support hand pose estimation using caffe
// By Hengkai Guo, Xinghao Chen

cv::Mat ReadDepthImageToCVMat(const string& filename, PoseDataParameter_Dataset dataset) {
    return ReadDepthImageToCVMat(filename, 0, 0, dataset);
}

cv::Mat ReadDepthImageToCVMat(const string& filename, const int height, const int width,
        PoseDataParameter_Dataset dataset) {
  // MSRA dataset
  if (dataset == PoseDataParameter_Dataset_MSRA) {
    if (filename.substr(filename.size() - 3) != "bin") {
        LOG(ERROR) << "Unsupported file extension for MSRA dataset: " << filename;
        return cv::Mat();
    }
    // Each bin file starts with 6 unsigned int: img_width img_height left top right bottom
    // see MSRA dataset for details
    FILE* in_file = fopen(filename.c_str(), "rb");
    unsigned int data_int[6];
    fread(data_int, sizeof(data_int[0]), sizeof(data_int) / sizeof(data_int[0]),
            in_file);
    // read depth data
    unsigned int total = (data_int[4] - data_int[2]) * (data_int[5] - data_int[3]);
    vector<float> data(total);
    fread(&data[0], sizeof(data[0]), data.size(), in_file);
    fclose(in_file);
    // convert to cv Mat
    // Note that all zero values are replaced with a maximum value
    // for simplicity of hand cropping
    cv::Mat depth = cv::Mat::ones(data_int[1], data_int[0], CV_32F) * 10000;
    int k = 0;
    for (unsigned int i = data_int[3]; i < data_int[5]; ++i)
        for (unsigned int j = data_int[2]; j < data_int[4]; ++j, ++k)
        {
            if (data[k] > 0)
            {
                depth.at<float>(i, j) = data[k];
            }
        }
    return depth;
  }
  // ICVL dataset and Hands17 dataset 
  else if (dataset == PoseDataParameter_Dataset_ICVL || dataset == PoseDataParameter_Dataset_HANDS17) {
    // depth images are stored in 16 bit grayscale png format 
    cv::Mat cv_img_origin = cv::imread(filename, CV_LOAD_IMAGE_ANYDEPTH);
    if (!cv_img_origin.data) {
      LOG(ERROR) << "Could not open or find file " << filename;
      return cv_img_origin;
    }
    cv::Mat cv_img;
    // If height and width are explicitly given, resize the depth image
    // In generally this is not needed
    if (height > 0 && width > 0) {
      cv::resize(cv_img_origin, cv_img, cv::Size(width, height));
    } else {
      cv_img = cv_img_origin;
    }
    // convert all data to float
    cv_img.convertTo(cv_img, CV_32F);
    // replace zero values with a maximum value
    cv_img.setTo(10000, cv_img == 0);
    return cv_img;
  } 
  // NYU dataset
  else if (dataset == PoseDataParameter_Dataset_NYU) {
    cv::Mat cv_image = ReadImageToCVMat(filename, height, width);
    cv::Mat depth(cv_image.rows, cv_image.cols, CV_32F);
    // In each depth png file the top 8 bits of depth are packed into the green channel and the lower 8 bits into blue.
    // See NYU dataset (https://cims.nyu.edu/~tompson/NYU_Hand_Pose_Dataset.htm#download) for details
    for (int r = 0; r < cv_image.rows; ++r) {
        const uchar* ori_ptr = cv_image.ptr<uchar>(r);
        float* ptr = depth.ptr<float>(r);
        for (int c = 0, n = 0; c < cv_image.cols; ++c, n += 3) {
            int val = ori_ptr[n + 1];
            val = (val << 8) + ori_ptr[n];
            ptr[c] = val;
        }
    }
    return depth;
  }
  // not supported dataset
  else {
    LOG(ERROR) << "Unsupported dataset: " << dataset;
    return cv::Mat();    
  }
}

void GetCenter(const cv::Mat& cv_img, vector<float>& center, int lower, int upper) {
    center = vector<float>(3, 0);
    int count = 0;
    // get pixels within depth range [lower, upper]
    for (int r = 0; r < cv_img.rows; ++r) {
        const float* ptr = cv_img.ptr<float>(r);
        for (int c = 0; c < cv_img.cols; ++c) {
            if (ptr[c] <= upper && ptr[c] >= lower) {
                center[0] += c;
                center[1] += r;
                center[2] += ptr[c];
                ++count;
            }
        }
    }
    // caculate center of mass
    if (count) {
        for (int i = 0; i < 3; ++i) {
            center[i] /= count;
        }
    }
    else
    {
        center.clear();
    }
}

void GetCenterByBboxFast(const cv::Mat& cv_img, vector<float> bbox, vector<float>& center, int lower, int upper) {
    center = vector<float>(3, 0);
    int count = 0;
    vector<float> center_naive = vector<float>(3, 0);
    int count_naive = 0;
    // get pixels within depth range [lower, upper] in bounding box
    // bbox: (c,r)_{topleft}, height, width
    for (int r = bbox[1]; r < std::min(bbox[1]+bbox[3], 1.0f*cv_img.rows); ++r) {
        const float* ptr = cv_img.ptr<float>(r);
        for (int c = bbox[0]; c < std::min(bbox[0]+bbox[2], 1.0f*cv_img.cols); ++c) {
            if (ptr[c] <= upper && ptr[c] >= lower) {
                center[0] += c;
                center[1] += r;
                center[2] += ptr[c];
                ++count;
            }
            // in case that depth values within bbox are out of range [lower, upper]
            // caculate a naive center in range [lower, 2048]
            // it's not optimal and can be further improved
            if (ptr[c] <= 2048 && ptr[c] >= lower) {
                center_naive[0] += c;
                center_naive[1] += r;
                center_naive[2] += ptr[c];
                ++count_naive;
            }
        }
    }
    if (count) {
        for (int i = 0; i < 3; ++i) {
            center[i] /= count;
        }
    }
    // if not getting center, use naive center instead
    else if(count_naive) {
        for (int i = 0; i < 3; ++i) {
            center[i] = center_naive[i]/count_naive;
        }
    }
    else {
        center.clear();
    }
}

cv::Mat CropImage(const cv::Mat& cv_img, vector<vector<float> >& points,
        const vector<float>& center, const vector<int>& cube_length,
        float fx, float fy, int height, int width) {
    // caculate the cropping rectangle by projecting 3D cube in image plane
    float xstart = center[0] - cube_length[0] / center[2] * fabs(fx);
    float xend = center[0] + cube_length[0] / center[2] * fabs(fx);
    float ystart = center[1] - cube_length[1] / center[2] * fabs(fy);
    float yend = center[1] + cube_length[1] / center[2] * fabs(fy);
    float xscale = 2.0 / (xend - xstart);
    float yscale = 2.0 / (yend - ystart);
    // points: hand joints
    // normalize the hand pose using the center and 3D cube
    for (size_t i = 0; i < points.size(); ++i) {
        points[i][0] -= xstart;
        points[i][0] *= xscale;
        points[i][0] -= 1;  // -1 ~ 1
        points[i][1] -= ystart;
        points[i][1] *= yscale;
        points[i][1] -= 1;  // -1 ~ 1
        points[i][2] -= center[2];
        points[i][2] /= cube_length[2];  // -1 ~ 1
    }
    // setup transformation matrix
    vector<cv::Point2f> src, dst;
    src.push_back(cv::Point2f(xstart, ystart));
    dst.push_back(cv::Point2f(0, 0));
    src.push_back(cv::Point2f(xstart, yend));
    dst.push_back(cv::Point2f(0, height - 1));
    src.push_back(cv::Point2f(xend, ystart));
    dst.push_back(cv::Point2f(width - 1, 0));
    cv::Mat trans = cv::getAffineTransform(src, dst);
    // get cropped hand image
    cv::Mat res_img;
    cv::warpAffine(cv_img, res_img, trans, cv::Size(width, height),
            cv::INTER_LINEAR, cv::BORDER_CONSTANT, center[2] + cube_length[2]);
    // normalize cropped image to [-1, 1]
    res_img -= center[2];
    res_img = cv::max(res_img, -cube_length[2]);
    res_img = cv::min(res_img, cube_length[2]);
    res_img /= cube_length[2];
    return res_img;
}

cv::Mat CropImageInBbox(const cv::Mat& cv_img, vector<float> bbox, vector<vector<float> >& points,
        const vector<float>& center, const vector<int>& cube_length,
        float fx, float fy, int height, int width) {
    // caculate the cropping rectangle by projecting 3D cube in image plane
    float xstart = center[0] - cube_length[0] / center[2] * fabs(fx);
    float xend = center[0] + cube_length[0] / center[2] * fabs(fx);
    float ystart = center[1] - cube_length[1] / center[2] * fabs(fy);
    float yend = center[1] + cube_length[1] / center[2] * fabs(fy);
    float xscale = 2.0 / (xend - xstart);
    float yscale = 2.0 / (yend - ystart);

    // only remain pixels within the bounding box
    cv::Mat depth = cv::Mat::ones(cv_img.rows, cv_img.cols, CV_32F) * 10000;
    cv::Rect roi = cv::Rect(bbox[0], bbox[1], bbox[2], bbox[3]);
    cv::Mat img_roi(depth, roi);
    cv_img(roi).copyTo(img_roi);

    // points: hand joints
    // normalize the hand pose using the center and 3D cube
    for (size_t i = 0; i < points.size(); ++i) {
        points[i][0] -= xstart;
        points[i][0] *= xscale;
        points[i][0] -= 1;  // -1 ~ 1
        points[i][1] -= ystart;
        points[i][1] *= yscale;
        points[i][1] -= 1;  // -1 ~ 1
        points[i][2] -= center[2];
        points[i][2] /= cube_length[2];  // -1 ~ 1
    }
    // setup transformation matrix
    vector<cv::Point2f> src, dst;
    src.push_back(cv::Point2f(xstart, ystart));
    dst.push_back(cv::Point2f(0, 0));
    src.push_back(cv::Point2f(xstart, yend));
    dst.push_back(cv::Point2f(0, height - 1));
    src.push_back(cv::Point2f(xend, ystart));
    dst.push_back(cv::Point2f(width - 1, 0));
    cv::Mat trans = cv::getAffineTransform(src, dst);
    // get cropped hand image
    cv::Mat res_img;
    cv::warpAffine(depth, res_img, trans, cv::Size(width, height),
            cv::INTER_LINEAR, cv::BORDER_CONSTANT, center[2] + cube_length[2]);
    // normalize cropped image to [-1, 1]
    res_img -= center[2];
    res_img = cv::max(res_img, -cube_length[2]);
    res_img = cv::min(res_img, cube_length[2]);
    res_img /= cube_length[2];
    return res_img;
}

// --------------------------------------------------------------------

#endif  // USE_OPENCV

bool ReadFileToDatum(const string& filename, const int label,
    Datum* datum) {
  std::streampos size;

  fstream file(filename.c_str(), ios::in|ios::binary|ios::ate);
  if (file.is_open()) {
    size = file.tellg();
    std::string buffer(size, ' ');
    file.seekg(0, ios::beg);
    file.read(&buffer[0], size);
    file.close();
    datum->set_data(buffer);
    datum->set_label(label);
    datum->set_encoded(true);
    return true;
  } else {
    return false;
  }
}

#ifdef USE_OPENCV
cv::Mat DecodeDatumToCVMatNative(const Datum& datum) {
  cv::Mat cv_img;
  CHECK(datum.encoded()) << "Datum not encoded";
  const string& data = datum.data();
  std::vector<char> vec_data(data.c_str(), data.c_str() + data.size());
  cv_img = cv::imdecode(vec_data, -1);
  if (!cv_img.data) {
    LOG(ERROR) << "Could not decode datum ";
  }
  return cv_img;
}
cv::Mat DecodeDatumToCVMat(const Datum& datum, bool is_color) {
  cv::Mat cv_img;
  CHECK(datum.encoded()) << "Datum not encoded";
  const string& data = datum.data();
  std::vector<char> vec_data(data.c_str(), data.c_str() + data.size());
  int cv_read_flag = (is_color ? CV_LOAD_IMAGE_COLOR :
    CV_LOAD_IMAGE_GRAYSCALE);
  cv_img = cv::imdecode(vec_data, cv_read_flag);
  if (!cv_img.data) {
    LOG(ERROR) << "Could not decode datum ";
  }
  return cv_img;
}

// If Datum is encoded will decoded using DecodeDatumToCVMat and CVMatToDatum
// If Datum is not encoded will do nothing
bool DecodeDatumNative(Datum* datum) {
  if (datum->encoded()) {
    cv::Mat cv_img = DecodeDatumToCVMatNative((*datum));
    CVMatToDatum(cv_img, datum);
    return true;
  } else {
    return false;
  }
}
bool DecodeDatum(Datum* datum, bool is_color) {
  if (datum->encoded()) {
    cv::Mat cv_img = DecodeDatumToCVMat((*datum), is_color);
    CVMatToDatum(cv_img, datum);
    return true;
  } else {
    return false;
  }
}

void CVMatToDatum(const cv::Mat& cv_img, Datum* datum) {
  CHECK(cv_img.depth() == CV_8U) << "Image data type must be unsigned byte";
  datum->set_channels(cv_img.channels());
  datum->set_height(cv_img.rows);
  datum->set_width(cv_img.cols);
  datum->clear_data();
  datum->clear_float_data();
  datum->set_encoded(false);
  int datum_channels = datum->channels();
  int datum_height = datum->height();
  int datum_width = datum->width();
  int datum_size = datum_channels * datum_height * datum_width;
  std::string buffer(datum_size, ' ');
  for (int h = 0; h < datum_height; ++h) {
    const uchar* ptr = cv_img.ptr<uchar>(h);
    int img_index = 0;
    for (int w = 0; w < datum_width; ++w) {
      for (int c = 0; c < datum_channels; ++c) {
        int datum_index = (c * datum_height + h) * datum_width + w;
        buffer[datum_index] = static_cast<char>(ptr[img_index++]);
      }
    }
  }
  datum->set_data(buffer);
}
#endif  // USE_OPENCV
}  // namespace caffe
