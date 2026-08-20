#pragma once

#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

// OCR untuk membaca teks plat nomor dari crop gambar
// Menggunakan model PaddleOCR yang sudah di-export ke ONNX
// Model ini hanya membaca teks, deteksi lokasi plat dilakukan oleh YoloDetector
class PlateTextRecognizer {
 public:
  PlateTextRecognizer(Ort::Env& env, const std::string& model_path,
                       const std::string& char_dict_path, int input_height = 48);

  // Input crop harus dalam format BGR (standar OpenCV)
  std::string recognize(const cv::Mat& crop);

 private:
  Ort::Session session_;
  Ort::AllocatorWithDefaultOptions allocator_;
  Ort::MemoryInfo mem_info_;
  std::string input_name_str_;
  std::string output_name_str_;
  const char* input_names_[1];
  const char* output_names_[1];

  int input_height_;
  std::vector<std::string> char_dict_;

  void loadCharDict(const std::string& path);
  cv::Mat preprocess(const cv::Mat& crop, int& out_width) const;
  std::string ctcGreedyDecode(float* output_data, const std::vector<int64_t>& output_shape) const;
};
