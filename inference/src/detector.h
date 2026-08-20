#pragma once

#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

// Hasil deteksi: box sudah dalam koordinat asli frame, bukan koordinat letterbox
struct Detection {
  cv::Rect box;
  float score;
  int class_id;
};

struct LetterboxInfo {
  float scale;
  int pad_x;
  int pad_y;
};

// Wrapper ONNX untuk model YOLO dari Ultralytics
// Bisa dipakai untuk model1 (deteksi kendaraan), plate (deteksi plat), dan model2 (deteksi helm)
// Class ini fokus ke inferensi saja, logika tracking dan caching ada di main.cpp
class YoloDetector {
 public:
  // class_names harus urut sesuai dengan training (cek data.yaml)
  YoloDetector(Ort::Env& env, const std::string& model_path, int num_classes,
               std::vector<std::string> class_names, float conf_thresh = 0.25f,
               float nms_thresh = 0.45f, int input_w = 640, int input_h = 640);

  std::vector<Detection> detect(const cv::Mat& frame);

  const std::string& className(int class_id) const;

 private:
  Ort::Session session_;
  Ort::AllocatorWithDefaultOptions allocator_;
  Ort::MemoryInfo mem_info_;
  
  // Simpan sebagai string biasa supaya kompatibel dengan berbagai versi ONNX Runtime
  std::string input_name_str_;
  std::string output_name_str_;
  const char* input_names_[1];
  const char* output_names_[1];

  int num_classes_;
  std::vector<std::string> class_names_;
  float conf_thresh_;
  float nms_thresh_;
  int input_w_;
  int input_h_;

  cv::Mat letterbox(const cv::Mat& src, LetterboxInfo& info) const;
  void preprocess(const cv::Mat& frame, std::vector<float>& input_tensor,
                   LetterboxInfo& lb_info) const;
  std::vector<Detection> decode(float* output_data, const std::vector<int64_t>& output_shape,
                                 const LetterboxInfo& lb_info) const;
};
