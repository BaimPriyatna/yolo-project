#include "detector.h"

#include <algorithm>

YoloDetector::YoloDetector(Ort::Env& env, const std::string& model_path, int num_classes,
                           std::vector<std::string> class_names, float conf_thresh,
                           float nms_thresh, int input_w, int input_h)
    : session_(nullptr),
      mem_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)),
      num_classes_(num_classes),
      class_names_(std::move(class_names)),
      conf_thresh_(conf_thresh),
      nms_thresh_(nms_thresh),
      input_w_(input_w),
      input_h_(input_h) {
  Ort::SessionOptions session_options;
  session_options.SetIntraOpNumThreads(0);  // auto-detect jumlah thread
  session_ = Ort::Session(env, model_path.c_str(), session_options);

  // Ambil nama input/output dari model dan simpan sebagai string
  auto input_name_alloc = session_.GetInputNameAllocated(0, allocator_);
  input_name_str_ = input_name_alloc.get();
  auto output_name_alloc = session_.GetOutputNameAllocated(0, allocator_);
  output_name_str_ = output_name_alloc.get();
  input_names_[0] = input_name_str_.c_str();
  output_names_[0] = output_name_str_.c_str();

  if (static_cast<int>(class_names_.size()) != num_classes_) {
    throw std::runtime_error("YoloDetector: jumlah class_names (" +
                              std::to_string(class_names_.size()) +
                              ") tidak sama dengan num_classes (" +
                              std::to_string(num_classes_) + ") untuk model: " + model_path);
  }
}

const std::string& YoloDetector::className(int class_id) const {
  return class_names_.at(class_id);
}

int YoloDetector::classIndexByName(const std::string& name) const {
  auto it = std::find(class_names_.begin(), class_names_.end(), name);
  if (it == class_names_.end()) {
    throw std::runtime_error("YoloDetector::classIndexByName: class '" + name +
                             "' tidak ditemukan di daftar class model ini.");
  }
  return static_cast<int>(std::distance(class_names_.begin(), it));
}

cv::Mat YoloDetector::letterbox(const cv::Mat& src, LetterboxInfo& info) const {
  float r = std::min(static_cast<float>(input_w_) / src.cols,
                      static_cast<float>(input_h_) / src.rows);
  int new_w = static_cast<int>(std::round(src.cols * r));
  int new_h = static_cast<int>(std::round(src.rows * r));

  cv::Mat resized;
  cv::resize(src, resized, cv::Size(new_w, new_h));

  cv::Mat out(input_h_, input_w_, CV_8UC3, cv::Scalar(114, 114, 114));
  int pad_x = (input_w_ - new_w) / 2;
  int pad_y = (input_h_ - new_h) / 2;
  resized.copyTo(out(cv::Rect(pad_x, pad_y, new_w, new_h)));

  info.scale = r;
  info.pad_x = pad_x;
  info.pad_y = pad_y;
  return out;
}

void YoloDetector::preprocess(const cv::Mat& frame, std::vector<float>& input_tensor,
                              LetterboxInfo& lb_info) const {
  cv::Mat letterboxed = letterbox(frame, lb_info);

  cv::Mat rgb;
  cv::cvtColor(letterboxed, rgb, cv::COLOR_BGR2RGB);
  rgb.convertTo(rgb, CV_32FC3, 1.0f / 255.0f);

  input_tensor.resize(1 * 3 * input_h_ * input_w_);
  std::vector<cv::Mat> channels(3);
  for (int c = 0; c < 3; ++c) {
    channels[c] = cv::Mat(input_h_, input_w_, CV_32FC1,
                          input_tensor.data() + c * input_h_ * input_w_);
  }
  cv::split(rgb, channels);
}

std::vector<Detection> YoloDetector::decode(float* output_data,
                                            const std::vector<int64_t>& output_shape,
                                            const LetterboxInfo& lb_info) const {
  int num_boxes = static_cast<int>(output_shape[2]);

  std::vector<cv::Rect> boxes;
  std::vector<float> scores;
  std::vector<int> class_ids;

  for (int i = 0; i < num_boxes; ++i) {
    float cx = output_data[0 * num_boxes + i];
    float cy = output_data[1 * num_boxes + i];
    float w = output_data[2 * num_boxes + i];
    float h = output_data[3 * num_boxes + i];

    float best_score = 0.0f;
    int best_class = -1;
    for (int c = 0; c < num_classes_; ++c) {
      float score = output_data[(4 + c) * num_boxes + i];
      if (score > best_score) {
        best_score = score;
        best_class = c;
      }
    }

    if (best_score < conf_thresh_) continue;

    float x1 = (cx - w / 2.0f - lb_info.pad_x) / lb_info.scale;
    float y1 = (cy - h / 2.0f - lb_info.pad_y) / lb_info.scale;
    float box_w = w / lb_info.scale;
    float box_h = h / lb_info.scale;

    boxes.emplace_back(cv::Rect(static_cast<int>(x1), static_cast<int>(y1),
                                static_cast<int>(box_w), static_cast<int>(box_h)));
    scores.push_back(best_score);
    class_ids.push_back(best_class);
  }

  std::vector<int> keep_indices;
  cv::dnn::NMSBoxes(boxes, scores, conf_thresh_, nms_thresh_, keep_indices);

  std::vector<Detection> detections;
  detections.reserve(keep_indices.size());
  for (int idx : keep_indices) {
    detections.push_back({boxes[idx], scores[idx], class_ids[idx]});
  }
  return detections;
}

std::vector<Detection> YoloDetector::detect(const cv::Mat& frame) {
  if (frame.empty()) return {};

  std::vector<float> input_tensor;
  LetterboxInfo lb_info;
  preprocess(frame, input_tensor, lb_info);

  std::array<int64_t, 4> input_shape = {1, 3, input_h_, input_w_};
  Ort::Value input_tensor_ort = Ort::Value::CreateTensor<float>(
      mem_info_, input_tensor.data(), input_tensor.size(), input_shape.data(),
      input_shape.size());

  auto output_tensors = session_.Run(Ort::RunOptions{nullptr}, input_names_, &input_tensor_ort,
                                     1, output_names_, 1);

  float* output_data = output_tensors[0].GetTensorMutableData<float>();
  auto output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();

  return decode(output_data, output_shape, lb_info);
}
