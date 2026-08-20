#include "recognizer.h"

#include <fstream>
#include <stdexcept>

PlateTextRecognizer::PlateTextRecognizer(Ort::Env& env, const std::string& model_path,
                                         const std::string& char_dict_path, int input_height)
    : session_(nullptr),
      mem_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)),
      input_height_(input_height) {
  loadCharDict(char_dict_path);

  Ort::SessionOptions session_options;
  session_options.SetIntraOpNumThreads(0);
  session_ = Ort::Session(env, model_path.c_str(), session_options);

  auto input_name_alloc = session_.GetInputNameAllocated(0, allocator_);
  input_name_str_ = input_name_alloc.get();
  auto output_name_alloc = session_.GetOutputNameAllocated(0, allocator_);
  output_name_str_ = output_name_alloc.get();
  input_names_[0] = input_name_str_.c_str();
  output_names_[0] = output_name_str_.c_str();

  // Validasi: jumlah class di model harus cocok dengan char_dict
  auto output_type_info = session_.GetOutputTypeInfo(0);
  auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
  int64_t num_classes = output_tensor_info.GetShape().back();
  int64_t expected = static_cast<int64_t>(char_dict_.size()) + 2;
  if (num_classes > 0 && num_classes != expected) {
    throw std::runtime_error(
        "PlateTextRecognizer: jumlah class output model (" + std::to_string(num_classes) +
        ") tidak cocok dengan char_dict.txt (" + std::to_string(char_dict_.size()) +
        " baris, ekspektasi " + std::to_string(expected) + " class)");
  }
}

void PlateTextRecognizer::loadCharDict(const std::string& path) {
  std::ifstream file(path);
  if (!file.is_open()) {
    throw std::runtime_error("PlateTextRecognizer: gagal membuka file " + path);
  }
  std::string line;
  while (std::getline(file, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    char_dict_.push_back(line);
  }
  if (char_dict_.empty()) {
    throw std::runtime_error("PlateTextRecognizer: file dictionary kosong " + path);
  }
}

cv::Mat PlateTextRecognizer::preprocess(const cv::Mat& crop, int& out_width) const {
  // Resize tinggi ke 48px, lebar menyesuaikan aspek rasio
  float ratio = static_cast<float>(crop.cols) / static_cast<float>(crop.rows);
  out_width = std::max(1, static_cast<int>(std::round(input_height_ * ratio)));

  cv::Mat resized;
  cv::resize(crop, resized, cv::Size(out_width, input_height_), 0, 0, cv::INTER_CUBIC);
  return resized;
}

std::string PlateTextRecognizer::ctcGreedyDecode(float* output_data,
                                                  const std::vector<int64_t>& output_shape) const {
  int64_t seq_len = output_shape[1];
  int64_t num_classes = output_shape[2];

  int last_class = -1;
  std::string result;

  for (int64_t t = 0; t < seq_len; ++t) {
    float* step = output_data + t * num_classes;
    int best_class = 0;
    float best_score = step[0];
    for (int64_t c = 1; c < num_classes; ++c) {
      if (step[c] > best_score) {
        best_score = step[c];
        best_class = static_cast<int>(c);
      }
    }

    if (best_class != 0 && best_class != last_class) {
      if (best_class == num_classes - 1) {
        result += ' ';
      } else {
        size_t dict_idx = static_cast<size_t>(best_class - 1);
        if (dict_idx < char_dict_.size()) {
          result += char_dict_[dict_idx];
        }
      }
    }
    last_class = best_class;
  }
  return result;
}

std::string PlateTextRecognizer::recognize(const cv::Mat& crop) {
  if (crop.empty() || crop.cols < 2 || crop.rows < 2) return "";

  // Plat nomor Indonesia punya 2 baris: baris 1 nopol utama, baris 2 tanggal berlaku
  // Kita crop 70% atas saja supaya OCR fokus ke nopol dan tidak terganggu baris bawah
  cv::Mat target_crop = crop;
  if (crop.rows >= 10) {
    int main_h = static_cast<int>(std::round(crop.rows * 0.70f));
    target_crop = crop(cv::Rect(0, 0, crop.cols, std::min(main_h, crop.rows)));
  }

  int width;
  cv::Mat resized = preprocess(target_crop, width);

  // Normalisasi (pixel/255 - 0.5) / 0.5, tanpa konversi BGR ke RGB
  cv::Mat float_img;
  resized.convertTo(float_img, CV_32FC3, 1.0 / 255.0);
  float_img = (float_img - 0.5) / 0.5;

  std::vector<float> input_tensor(1 * 3 * input_height_ * width);
  std::vector<cv::Mat> channels(3);
  for (int c = 0; c < 3; ++c) {
    channels[c] =
        cv::Mat(input_height_, width, CV_32FC1, input_tensor.data() + c * input_height_ * width);
  }
  cv::split(float_img, channels);

  std::array<int64_t, 4> input_shape = {1, 3, input_height_, width};
  Ort::Value input_tensor_ort = Ort::Value::CreateTensor<float>(
      mem_info_, input_tensor.data(), input_tensor.size(), input_shape.data(), input_shape.size());

  auto output_tensors = session_.Run(Ort::RunOptions{nullptr}, input_names_, &input_tensor_ort, 1,
                                     output_names_, 1);

  float* output_data = output_tensors[0].GetTensorMutableData<float>();
  auto output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();

  return ctcGreedyDecode(output_data, output_shape);
}
