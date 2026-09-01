#pragma once

#include <string>

// Semua angka threshold yang biasanya perlu di-tuning pas kalibrasi, dikumpulin di 1 tempat.
// Dibaca dari file teks key=value (../models/pipeline_config.txt) SAAT RUNTIME - beda nilai
// buat dicoba = edit file ini, TIDAK PERLU recompile main.cpp.
//
// File config ini OPSIONAL: kalau nggak ada, semua field pakai nilai default di bawah
// (behaviour sama persis kayak sebelum fitur ini ada) - jadi nggak akan bikin pipeline
// gagal jalan cuma gara-gara lupa bikin file config.
struct PipelineConfig {
  float model1_conf_thresh = 0.25f;
  float model1_nms_thresh = 0.45f;

  float plate_conf_thresh = 0.25f;
  float plate_nms_thresh = 0.45f;

  float model2_conf_thresh = 0.25f;
  float model2_nms_thresh = 0.45f;

  int bytetrack_track_buffer = 30;
  float bytetrack_track_thresh = 0.5f;
  float bytetrack_high_thresh = 0.6f;
  float bytetrack_match_thresh = 0.8f;

  // Padding di sekitar box plat sebelum di-crop buat OCR (rasio dari lebar/tinggi box plat)
  float plate_padding_ratio = 0.04f;

  // Minimum IoU antara box person & box motor buat dianggap "driver". Default 0.0 berarti
  // exact perilaku lama (asal ada overlap sedikit pun, dianggap driver).
  float min_driver_overlap_iou = 0.0f;
};

// Load dari file key=value (format: "nama_key=angka", baris kosong/diawali '#' diabaikan).
// Kalau file nggak ada atau ada baris error, print warning ke stderr dan LANJUT pakai
// default utk key yang gagal dibaca (bukan crash) - config ini sengaja "best-effort".
PipelineConfig loadPipelineConfig(const std::string& path);
