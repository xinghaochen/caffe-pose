/*************************************************************************
    > File Name: output_pose.cpp
    > Author: Guo Hengkai
    > Description: 
    > Created Time: Sun 06 Nov 2016 10:47:29 AM CST
 ************************************************************************/
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <gflags/gflags.h>
#include <glog/logging.h>

#include "boost/algorithm/string.hpp"
#include "google/protobuf/text_format.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/net.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/format.hpp"
#include "caffe/util/io.hpp"

using caffe::Blob;
using caffe::Caffe;
using caffe::Datum;
using caffe::Net;
using std::string;
using std::vector;

DEFINE_int32(gpu, -1,
    "Optional; run in GPU mode on given device ID. -1 for CPU.");
DEFINE_string(model, "",
    "The model definition protocol buffer text file.");
DEFINE_string(weights, "",
    "The weights to initialize model.");
DEFINE_string(phase, "",
    "Network phase (train or test).");
DEFINE_string(label_list, "",
    "Ground truth list.");
DEFINE_string(error_blob, "error",
    "Blob name for error.");
DEFINE_string(output_blob, "output",
    "Blob name for output.");
DEFINE_string(output_name, "",
    "Output file name, per pose per line.");
DEFINE_double(fx, 588.03, "fx");
DEFINE_double(fy, 587.07, "fy");
DEFINE_double(ux, 320, "ux");
DEFINE_double(uy, 240, "uy");
DEFINE_double(th, 20, "th");

#define OUTPUT_CENTER 0

#define SQR(x) ((x)*(x))
float GetDistance(const vector<float>& res, const vector<float>& lbl, vector<float>& detect,
        vector<int>& count) {
    float dis = 0;
    for (int i = 0; i < res.size(); i += 3) {
        float u1 = res[i];
        float v1 = res[i + 1];
        float d1 = res[i + 2];
        float x1 = (u1 - FLAGS_ux) * d1 / FLAGS_fx;
        float y1 = (v1 - FLAGS_uy) * d1 / FLAGS_fy;
        float u2 = lbl[i];
        float v2 = lbl[i + 1];
        float d2 = lbl[i + 2];
        if (fabs(u2 + v2 + d2) < 1e-2) {
            continue;
        }
        float x2 = (u2 - FLAGS_ux) * d2 / FLAGS_fx;
        float y2 = (v2 - FLAGS_uy) * d2 / FLAGS_fy;
        float d = sqrt(SQR(x1 - x2) + SQR(y1 - y2) + SQR(d1 - d2));
        dis += d;
        ++count[i / 3];
        if (d <= FLAGS_th) {
            ++detect[i / 3];
        }
    }
    return dis;
}

int main(int argc, char** argv) {
    FLAGS_alsologtostderr = 1;
    ::google::InitGoogleLogging(argv[0]);
    ::google::ParseCommandLineFlags(&argc, &argv, true);
    CHECK(!FLAGS_model.empty()) << "model cannot be empty";
    CHECK(!FLAGS_label_list.empty()) << "label_list cannot be empty";

    // load labels
    vector<vector<float> > labels;
    std::ifstream infile(FLAGS_label_list.c_str());
    CHECK(infile.good()) << "Fail to open the file " << FLAGS_label_list;
    LOG(INFO) << "Reading labels from " << FLAGS_label_list;
    string line;
    while (std::getline(infile, line)) {
        std::istringstream is(line);
        vector<float> label;
        float x;
        while (is >> x) {
            label.push_back(x);
        }
        labels.push_back(label);
    }
    CHECK(!labels.empty()) << "no label loaded";
    size_t n = labels.size();
    LOG(INFO) << "label size: " << n << " * " << labels[0].size();

    // initialize model
    if (FLAGS_gpu < 0) {
        LOG(INFO) << "Using CPU";
        Caffe::set_mode(Caffe::CPU);
    } else {
        LOG(INFO) << "Using GPU " << FLAGS_gpu;
        Caffe::SetDevice(FLAGS_gpu);
        Caffe::set_mode(Caffe::GPU);
    }
    boost::shared_ptr<Net<float> > test_net(
        new Net<float>(FLAGS_model, caffe::TEST));
    if (!FLAGS_weights.empty()) {
        test_net->CopyTrainedLayersFrom(FLAGS_weights);
    } else {
        LOG(WARNING) << "weight is empty!";
    }

    // get outputs and save
    vector<vector<float> > results;
    const boost::shared_ptr<Blob<float> > blob =
        test_net->blob_by_name(FLAGS_output_blob);
    const boost::shared_ptr<Blob<float> > error_blob =
        test_net->blob_by_name(FLAGS_error_blob);
#if OUTPUT_CENTER
    std::ofstream outfile_center("center.txt");
    const boost::shared_ptr<Blob<float> > center_blob =
        test_net->blob_by_name("center");
#endif
    int batch_size = blob->num();
    int res_size = blob->count() / batch_size;
    LOG(INFO) << "batch size: " << batch_size;
    int batch_num = (n - 1) / batch_size + 1;
    int remain = n - batch_size * (batch_num - 1);
    std::ofstream outfile(FLAGS_output_name.c_str());
    CHECK(outfile.good()) << "Fail to open the file " << FLAGS_output_name;
    LOG(INFO) << "Writing into " << FLAGS_output_name;
    for (int i = 0; i < batch_num; ++i) {
        test_net->Forward();
        LOG(INFO) << "Error for batch " << i << ": "
            << error_blob->cpu_data()[0] << "mm";
        int m = (i == batch_num - 1) ? remain : batch_size;
        const float* data = blob->cpu_data();
#if OUTPUT_CENTER
        const float* center = center_blob->cpu_data();
#endif
        for (int j = 0; j < m; ++j) {
            vector<float> result;
            for (int k = 0; k < res_size; ++k) {
                result.push_back(*data);
                outfile << *data << " ";
                ++data;
            }
#if OUTPUT_CENTER
            for (int k = 0; k < 3; ++k) {
                outfile_center << *center << " ";
                ++center;
            }
            outfile_center << std::endl;
#endif
            results.push_back(result);
            outfile << std::endl;
        }
    }
    CHECK_EQ(results.size(), n);

    // calculate the errors and std
    vector<float> sum(results[0].size(), 0);
    vector<float> sum_sqr(results[0].size(), 0);
    for (size_t i = 0; i < n; ++i) {
      for (size_t j = 0; j < results[i].size(); ++j) {
        float delta = results[i][j] - labels[i][j];
        sum[j] += delta;
        sum_sqr[j] += delta * delta;
      }
    }
    vector<float> stds(sum.size());
    vector<float> mean_stds(3, 0.0);
    for (size_t j = 0; j < sum.size(); ++j) {
      stds[j] = sqrt((sum_sqr[j] - sum[j] * sum[j] / n) / n);
      mean_stds[j % 3] += stds[j];
    }
    for (int j = 0; j < 3; ++j) {
        mean_stds[j] /= (sum.size() / 3);
    }
    LOG(INFO) << "Average std: " << mean_stds[0] << ", " << mean_stds[1]
        << ", " << mean_stds[2];

    float error = 0.0f;
    vector<int> count(results[0].size() / 3, 0);
    vector<float> detect(results[0].size() / 3, 0);
    for (int i = 0; i < n; ++i) {
        error += GetDistance(results[i], labels[i], detect, count);
    }
    int total = 0;
    float map = 0.0;
    for (size_t i = 0; i < count.size(); ++i) {
        total += count[i];
        detect[i] /= count[i];
        map += detect[i];
    }
    error /= total;
    map /= count.size();
    LOG(INFO) << "Error on dataset: " << error << "mm";
    LOG(INFO) << "mAP on dataset: " << map * 100 << "% (th = " << FLAGS_th << "mm)";
    return 0;
}
