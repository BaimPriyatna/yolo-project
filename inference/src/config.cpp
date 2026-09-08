#include "config.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <iostream>
#include <sstream>
#include <unordered_map>

namespace {

std::string trim(const std::string& s) {
  size_t start = s.find_first_not_of(" \t\r\n");
  if (start == std::string::npos) return "";
  size_t end = s.find_last_not_of(" \t\r\n");
  return s.substr(start, end - start + 1);
}

// Set field float/int dari map kalau key-nya ada, print warning kalau gagal parse angka.
void setFloatIfPresent(const std::unordered_map<std::string, std::string>& kv, const std::string& key,
                       float& out) {
  auto it = kv.find(key);
  if (it == kv.end()) return;
  try {
    out = std::stof(it->second);
  } catch (...) {
    std::cerr << "[config] Warning: nilai '" << it->second << "' untuk '" << key
              << "' bukan angka valid, pakai default (" << out << ")\n";
  }
}

void setIntIfPresent(const std::unordered_map<std::string, std::string>& kv, const std::string& key,
                     int& out) {
  auto it = kv.find(key);
  if (it == kv.end()) return;
  try {
    out = std::stoi(it->second);
  } catch (...) {
    std::cerr << "[config] Warning: nilai '" << it->second << "' untuk '" << key
              << "' bukan angka valid, pakai default (" << out << ")\n";
  }
}

}  // namespace

PipelineConfig loadPipelineConfig(const std::string& path) {
  PipelineConfig config;  // mulai dari semua default

  std::ifstream file(path);
  if (!file.is_open()) {
    std::cout << "[config] " << path << " tidak ditemukan, pakai nilai default semua threshold.\n"
              << "[config] (opsional - copy dari inference/config/pipeline_config.example.txt "
                 "kalau mau tuning)\n";
    return config;
  }

  std::unordered_map<std::string, std::string> kv;
  std::string line;
  int line_num = 0;
  while (std::getline(file, line)) {
    ++line_num;
    line = trim(line);
    if (line.empty() || line[0] == '#') continue;

    size_t eq_pos = line.find('=');
    if (eq_pos == std::string::npos) {
      std::cerr << "[config] Warning: baris " << line_num << " di " << path
                << " diabaikan (format harus key=value): " << line << "\n";
      continue;
    }
    std::string key = trim(line.substr(0, eq_pos));
    std::string value = trim(line.substr(eq_pos + 1));
    kv[key] = value;
  }

  setFloatIfPresent(kv, "model1_conf_thresh", config.model1_conf_thresh);
  setFloatIfPresent(kv, "model1_nms_thresh", config.model1_nms_thresh);
  setFloatIfPresent(kv, "plate_conf_thresh", config.plate_conf_thresh);
  setFloatIfPresent(kv, "plate_nms_thresh", config.plate_nms_thresh);
  setFloatIfPresent(kv, "model2_conf_thresh", config.model2_conf_thresh);
  setFloatIfPresent(kv, "model2_nms_thresh", config.model2_nms_thresh);
  setIntIfPresent(kv, "bytetrack_track_buffer", config.bytetrack_track_buffer);
  setFloatIfPresent(kv, "bytetrack_track_thresh", config.bytetrack_track_thresh);
  setFloatIfPresent(kv, "bytetrack_high_thresh", config.bytetrack_high_thresh);
  setFloatIfPresent(kv, "bytetrack_match_thresh", config.bytetrack_match_thresh);
  setFloatIfPresent(kv, "plate_padding_ratio", config.plate_padding_ratio);
  setFloatIfPresent(kv, "min_driver_overlap_iou", config.min_driver_overlap_iou);
  setIntIfPresent(kv, "frame_skip", config.frame_skip);
  setIntIfPresent(kv, "capture_width", config.capture_width);
  setIntIfPresent(kv, "capture_height", config.capture_height);

  std::cout << "[config] Berhasil load threshold dari " << path << "\n";
  return config;
}
