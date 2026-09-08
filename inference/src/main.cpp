#include <ByteTrack/BYTETracker.h>
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/filesystem.hpp>

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "config.h"
#include "detector.h"
#include "recognizer.h"

// Class id TIDAK di-hardcode. Nama class dibaca dari models/model1_classes.txt,
// index dicari via classIndexByName() saat runtime - ganti model (COCO <-> custom)
// = ganti isi file itu, tanpa recompile.
namespace Model2 {
constexpr int HELMET = 0;
constexpr int NO_HELMET = 1;
}

std::vector<std::string> loadClassNames(const std::string& path) {
  std::ifstream file(path);
  if (!file.is_open()) {
    throw std::runtime_error("Gagal buka file class list: " + path +
                             "\nBuat file ini (1 nama class per baris) sesuai model1.onnx. "
                             "Lihat template di inference/config/.");
  }
  std::vector<std::string> names;
  std::string line;
  while (std::getline(file, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (!line.empty()) names.push_back(line);
  }
  return names;
}

struct PlateCacheEntry {
  bool found = false;
  cv::Rect box;
  std::string text;
};

bool boxesOverlap(const cv::Rect& a, const cv::Rect& b, float min_iou = 0.0f) {
  if ((a & b).area() <= 0) return false;
  if (min_iou <= 0.0f) return true;  // sama persis perilaku lama: asal overlap sedikit pun, true
  float inter_area = static_cast<float>((a & b).area());
  float union_area = static_cast<float>(a.area() + b.area() - inter_area);
  float iou = (union_area > 0) ? (inter_area / union_area) : 0.0f;
  return iou >= min_iou;
}

float calcIoU(const cv::Rect& a, const cv::Rect& b) {
  float inter_area = static_cast<float>((a & b).area());
  float union_area = static_cast<float>(a.area() + b.area() - inter_area);
  return (union_area > 0) ? (inter_area / union_area) : 0.0f;
}

bool isNumber(const std::string& s) {
  return !s.empty() && std::all_of(s.begin(), s.end(), ::isdigit);
}

void drawLabeledBox(cv::Mat& frame, const cv::Rect& box, const std::string& label, const cv::Scalar& color) {
  cv::Rect clipped_box = box & cv::Rect(0, 0, frame.cols, frame.rows);
  if (clipped_box.width <= 0 || clipped_box.height <= 0) return;

  cv::rectangle(frame, clipped_box, color, 2);
  int baseline = 0;
  cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
  cv::Point text_org(clipped_box.x, std::max(clipped_box.y - 5, text_size.height));
  cv::rectangle(frame, text_org + cv::Point(0, baseline), text_org + cv::Point(text_size.width, -text_size.height),
                color, cv::FILLED);
  cv::putText(frame, label, text_org, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
}

// Pipeline untuk satu kendaraan: deteksi plat + OCR + cek helm (khusus motor)
struct VehicleClassIds {
  int person;
  int motorcycle;
  int car;
  int bus;
  int truck;
};

void processVehicle(cv::Mat& frame, const cv::Rect& vehicle_box, int class_id, int track_id,
                    const VehicleClassIds& ids, const PipelineConfig& config,
                    const std::vector<Detection>& person_dets, YoloDetector& plate_model,
                    YoloDetector& helmet_model, PlateTextRecognizer& plate_recognizer,
                    std::unordered_map<int, PlateCacheEntry>& plate_cache,
                    std::unordered_map<int, std::string>& helmet_cache) {
  cv::Rect clipped_box = vehicle_box & cv::Rect(0, 0, frame.cols, frame.rows);
  if (clipped_box.width <= 0 || clipped_box.height <= 0) return;

  cv::Scalar box_color = (class_id == ids.motorcycle) ? cv::Scalar(0, 255, 0) :
                         (class_id == ids.car)        ? cv::Scalar(255, 0, 0) :
                         (class_id == ids.bus)        ? cv::Scalar(200, 50, 0) :
                                                            cv::Scalar(180, 100, 0);
  cv::rectangle(frame, clipped_box, box_color, 2);
  cv::Mat vehicle_crop = frame(clipped_box);

  // Deteksi plat nomor dan OCR
  if (plate_cache.find(track_id) == plate_cache.end()) {
    auto plate_dets = plate_model.detect(vehicle_crop);
    PlateCacheEntry entry;
    std::vector<Detection> valid_plates;
    for (const auto& d : plate_dets) {
      if (d.class_id == 0) valid_plates.push_back(d);
    }
    if (!valid_plates.empty()) {
      auto best = *std::max_element(
          valid_plates.begin(), valid_plates.end(),
          [](const Detection& a, const Detection& b) { return a.score < b.score; });
      entry.found = true;
      entry.box = best.box & cv::Rect(0, 0, vehicle_crop.cols, vehicle_crop.rows);
      if (entry.box.width > 0 && entry.box.height > 0) {
        int pad_x = std::max(2, static_cast<int>(entry.box.width * config.plate_padding_ratio));
        int pad_y = std::max(2, static_cast<int>(entry.box.height * config.plate_padding_ratio));
        int px1 = std::max(0, entry.box.x - pad_x);
        int py1 = std::max(0, entry.box.y - pad_y);
        int px2 = std::min(vehicle_crop.cols, entry.box.x + entry.box.width + pad_x);
        int py2 = std::min(vehicle_crop.rows, entry.box.y + entry.box.height + pad_y);
        cv::Rect padded_box(px1, py1, px2 - px1, py2 - py1);

        cv::Mat plate_crop = vehicle_crop(padded_box);
        entry.text = plate_recognizer.recognize(plate_crop);
      }
    }
    plate_cache[track_id] = entry;
  }
  const auto& plate_info = plate_cache[track_id];

  if (plate_info.found && plate_info.box.width > 0 && plate_info.box.height > 0) {
    cv::Rect abs_plate_box(clipped_box.x + plate_info.box.x, clipped_box.y + plate_info.box.y,
                           plate_info.box.width, plate_info.box.height);
    cv::rectangle(frame, abs_plate_box, cv::Scalar(0, 255, 255), 2);
  }

  // Cek helm khusus untuk motor
  std::string helmet_suffix = "";
  if (class_id == ids.motorcycle) {
    bool has_driver = std::any_of(person_dets.begin(), person_dets.end(),
                                  [&](const Detection& p) { return boxesOverlap(p.box, clipped_box, config.min_driver_overlap_iou); });
    std::string helmet_status = "no_rider";
    if (has_driver) {
      auto it = helmet_cache.find(track_id);
      if (it == helmet_cache.end()) {
        auto helmet_dets = helmet_model.detect(vehicle_crop);
        if (!helmet_dets.empty()) {
          auto best = *std::max_element(
              helmet_dets.begin(), helmet_dets.end(),
              [](const Detection& a, const Detection& b) { return a.score < b.score; });
          helmet_status = (best.class_id == Model2::HELMET) ? "helmet" : "no_helmet";
        } else {
          helmet_status = "unknown";
        }
        helmet_cache[track_id] = helmet_status;
      } else {
        helmet_status = it->second;
      }
    }
    helmet_suffix = " [" + helmet_status + "]";
  }

  // Tampilkan label
  std::string vehicle_name = (class_id == ids.motorcycle) ? "Motor" :
                             (class_id == ids.car)        ? "Car" :
                             (class_id == ids.bus)        ? "Bus" : "Truck";
  std::string plate_label = plate_info.found
                                ? (plate_info.text.empty() ? "Plate:?" : "Plate:" + plate_info.text)
                                : "Plate:-";
  std::string label = vehicle_name + " ID:" + std::to_string(track_id) + " [" + plate_label + "]" + helmet_suffix;
  std::cout << "  [Vehicle " << vehicle_name << " ID:" << track_id << "] " << plate_label << helmet_suffix << std::endl;

  int baseline = 0;
  cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.55, 2, &baseline);
  cv::Point text_org(clipped_box.x, std::max(clipped_box.y - 8, text_size.height));
  cv::rectangle(frame, text_org + cv::Point(0, baseline), text_org + cv::Point(text_size.width, -text_size.height),
                box_color, cv::FILLED);
  cv::putText(frame, label, text_org, cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(255, 255, 255), 2);
}

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <image_path | video_path | camera_index>\n";
    return 1;
  }
  std::string source = argv[1];

  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "yolo-2stage");

  PipelineConfig config = loadPipelineConfig("../models/pipeline_config.txt");

  std::unique_ptr<YoloDetector> model1_ptr, plate_model_ptr, model2_ptr;
  std::unique_ptr<PlateTextRecognizer> plate_recognizer_ptr;
  VehicleClassIds ids{};
  try {
    std::vector<std::string> model1_classes = loadClassNames("../models/model1_classes.txt");
    std::cout << "Loading Model 1 (" << model1_classes.size() << " classes dari model1_classes.txt)...\n";
    model1_ptr = std::make_unique<YoloDetector>(env, "../models/model1.onnx",
                                                static_cast<int>(model1_classes.size()),
                                                model1_classes, config.model1_conf_thresh,
                                                config.model1_nms_thresh);

    ids.person = model1_ptr->classIndexByName("person");
    ids.motorcycle = model1_ptr->classIndexByName("motorcycle");
    ids.car = model1_ptr->classIndexByName("car");
    try { ids.bus = model1_ptr->classIndexByName("bus"); } catch (...) { ids.bus = -1; }
    try { ids.truck = model1_ptr->classIndexByName("truck"); } catch (...) { ids.truck = -1; }

    std::cout << "Loading Plate model (2 classes: plate, vehicle)...\n";
    plate_model_ptr = std::make_unique<YoloDetector>(env, "../models/plate.onnx", 2,
                                                      std::vector<std::string>{"plate", "vehicle"},
                                                      config.plate_conf_thresh, config.plate_nms_thresh);

    std::cout << "Loading Model 2 (helmet, no_helmet)...\n";
    model2_ptr = std::make_unique<YoloDetector>(env, "../models/model2.onnx", 2,
                                                std::vector<std::string>{"helmet", "no_helmet"},
                                                config.model2_conf_thresh, config.model2_nms_thresh);

    std::cout << "Loading Plate text recognizer (CTC)...\n";
    plate_recognizer_ptr = std::make_unique<PlateTextRecognizer>(
        env, "../models/plate_rec.onnx", "../models/char_dict.txt");
  } catch (const Ort::Exception& e) {
    std::cerr << "\nGagal load model ONNX: " << e.what() << "\n"
              << "Pastikan model1.onnx, plate.onnx, dan model2.onnx ada di folder models/\n";
    return 1;
  }
  YoloDetector& model1 = *model1_ptr;
  YoloDetector& plate_model = *plate_model_ptr;
  YoloDetector& model2 = *model2_ptr;
  PlateTextRecognizer& plate_recognizer = *plate_recognizer_ptr;

  std::unordered_map<int, PlateCacheEntry> plate_cache;
  std::unordered_map<int, std::string> helmet_cache;
  std::unordered_map<int, int> track_class_map;

  std::string ext;
  size_t dot = source.find_last_of('.');
  if (dot != std::string::npos) ext = source.substr(dot + 1);
  std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
  bool is_image = (ext == "jpg" || ext == "jpeg" || ext == "png" || ext == "bmp");

  if (is_image) {
    cv::Mat img = cv::imread(source);
    if (img.empty()) {
      std::cerr << "Failed to load image: " << source << "\n";
      return 1;
    }

    auto dets = model1.detect(img);
    std::vector<Detection> person_dets;
    std::vector<Detection> vehicle_dets;
    int car_count = 0;
    int bus_count = 0;
    int truck_count = 0;
    int moto_count = 0;

    for (const auto& d : dets) {
      if (d.class_id == ids.person) {
        person_dets.push_back(d);
        drawLabeledBox(img, d.box, cv::format("Person %.2f", d.score), cv::Scalar(0, 215, 255));
      } else if (d.class_id == ids.motorcycle) {
        moto_count++;
        vehicle_dets.push_back(d);
      } else if (d.class_id == ids.car) {
        car_count++;
        vehicle_dets.push_back(d);
      } else if (d.class_id == ids.bus) {
        bus_count++;
        vehicle_dets.push_back(d);
      } else if (d.class_id == ids.truck) {
        truck_count++;
        vehicle_dets.push_back(d);
      }
    }

    int idx = 0;
    for (const auto& v : vehicle_dets) {
      processVehicle(img, v.box, v.class_id, idx++, ids, config, person_dets, plate_model, model2, plate_recognizer,
                     plate_cache, helmet_cache);
    }

    cv::utils::fs::createDirectories("../data/output");
    size_t slash = source.find_last_of("/\\");
    std::string filename = (slash != std::string::npos) ? source.substr(slash + 1) : source;
    std::string out_path = "../data/output/" + filename;
    cv::imwrite(out_path, img);

    std::cout << "Saved " << out_path << " | cars=" << car_count << " buses=" << bus_count
              << " trucks=" << truck_count << " motorcycles=" << moto_count
              << " persons=" << person_dets.size() << "\n";
    return 0;
  }

  // Mode video atau kamera dengan ByteTrack
  cv::VideoCapture cap;
  if (isNumber(source)) {
    cap.open(std::stoi(source));
  } else {
    cap.open(source);
  }
  if (!cap.isOpened()) {
    std::cerr << "Failed to open video source: " << source << "\n";
    return 1;
  }
  cap.set(cv::CAP_PROP_BUFFERSIZE, 1);
  if (config.capture_width > 0) cap.set(cv::CAP_PROP_FRAME_WIDTH, config.capture_width);
  if (config.capture_height > 0) cap.set(cv::CAP_PROP_FRAME_HEIGHT, config.capture_height);

  int fps_hint = static_cast<int>(cap.get(cv::CAP_PROP_FPS));
  if (fps_hint <= 0) fps_hint = 30;

  byte_track::BYTETracker tracker(fps_hint, config.bytetrack_track_buffer, config.bytetrack_track_thresh, config.bytetrack_high_thresh, config.bytetrack_match_thresh);

  // Struct kecil buat nyimpen hasil deteksi kendaraan terakhir - dipakai ulang di
  // frame yang di-skip (lihat config.frame_skip) biar nggak perlu jalanin ulang
  // Model 1 + ByteTrack (paling berat) di SETIAP frame.
  struct TrackedVehicle {
    int track_id;
    cv::Rect box;
    int class_id;
  };
  std::vector<TrackedVehicle> last_tracked_vehicles;
  std::vector<Detection> last_person_dets;

  cv::Mat frame;
  int frame_counter = 0;
  while (true) {
    auto t0 = std::chrono::steady_clock::now();

    for (int i = 0; i < 4; ++i) {
      if (!cap.grab()) break;
    }
    cap.retrieve(frame);
    if (frame.empty()) break;

    ++frame_counter;
    bool do_detect = (config.frame_skip <= 1) || (frame_counter % config.frame_skip == 0);

    if (do_detect) {
      // Stage 1: Deteksi objek utama (paling berat - full frame, 640x640)
      auto dets = model1.detect(frame);
      std::vector<Detection> person_dets;
      std::vector<Detection> vehicle_dets;
      for (const auto& d : dets) {
        if (d.class_id == ids.person) {
          person_dets.push_back(d);
        } else if (d.class_id == ids.motorcycle || d.class_id == ids.car ||
                   d.class_id == ids.bus || d.class_id == ids.truck) {
          vehicle_dets.push_back(d);
        }
      }

      // ByteTrack untuk tracking kendaraan
      std::vector<byte_track::Object> track_objects;
      track_objects.reserve(vehicle_dets.size());
      for (const auto& d : vehicle_dets) {
        byte_track::Rect<float> rect(static_cast<float>(d.box.x), static_cast<float>(d.box.y),
                                     static_cast<float>(d.box.width),
                                     static_cast<float>(d.box.height));
        track_objects.emplace_back(rect, 0, d.score);
      }
      auto tracked = tracker.update(track_objects);

      last_tracked_vehicles.clear();
      for (const auto& strack : tracked) {
        int track_id = static_cast<int>(strack->getTrackId());
        const auto& r = strack->getRect();
        cv::Rect v_box(static_cast<int>(r.x()), static_cast<int>(r.y()),
                       static_cast<int>(r.width()), static_cast<int>(r.height()));

        auto it_cls = track_class_map.find(track_id);
        int class_id = ids.car;
        if (it_cls != track_class_map.end()) {
          class_id = it_cls->second;
        } else {
          float max_iou = 0.0f;
          for (const auto& d : vehicle_dets) {
            float iou = calcIoU(d.box, v_box);
            if (iou > max_iou) {
              max_iou = iou;
              class_id = d.class_id;
            }
          }
          track_class_map[track_id] = class_id;
        }
        last_tracked_vehicles.push_back({track_id, v_box, class_id});
      }
      last_person_dets = person_dets;
    }
    // Kalau do_detect == false: last_tracked_vehicles & last_person_dets dari frame
    // terakhir yang diproses tetap dipakai apa adanya (posisi box jadi agak "nge-lag"
    // dikit dibanding kondisi real, tapi Model 1 + ByteTrack nggak perlu jalan lagi).

    for (const auto& d : last_person_dets) {
      drawLabeledBox(frame, d.box, "Person", cv::Scalar(0, 215, 255));
    }
    for (const auto& v : last_tracked_vehicles) {
      processVehicle(frame, v.box, v.class_id, v.track_id, ids, config, last_person_dets,
                     plate_model, model2, plate_recognizer, plate_cache, helmet_cache);
    }

    auto t1 = std::chrono::steady_clock::now();
    double fps = 1000.0 / std::chrono::duration<double, std::milli>(t1 - t0).count();
    cv::putText(frame, cv::format("FPS: %.1f | vehicles: %zu", fps, last_tracked_vehicles.size()), cv::Point(10, 25),
               cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);

    cv::imshow("2-Stage Traffic Pipeline", frame);
    int key = cv::waitKey(1);
    if (key == 'q' || key == 27) break;
  }

  cap.release();
  cv::destroyAllWindows();
  return 0;
}
