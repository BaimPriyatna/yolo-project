# Implementation Plan - Upgrade Model 1 & Fix Inference Pipeline

Perbaikan pipeline deteksi objek: mengganti Model 1 dengan model pre-trained standard COCO (YOLO11n 80 kelas) untuk mendeteksi kendaraan (mobil, motor, bus, truk) dan orang secara akurat, memperbaiki visualisasi bounding box & label, serta membetulkan direktori penyimpanan output.

## User Review Required

> [!IMPORTANT]
> - Model 2 (`model2.onnx` untuk helm/no_helmet), deteksi plat (`plate.onnx`), dan OCR plat (`plate_rec.onnx`) **tidak diubah** dan tetap berjalan pada stage kedua (crop motor).
> - Model 1 diganti ke pre-trained COCO YOLO11n (80 kelas) agar deteksi `car`, `bus`, `truck`, `motorcycle`, dan `person` jauh lebih robust dan sensitif pada wide-angle CCTV/dashcam.

---

## Proposed Changes

### 1. Model Preparation & Export
- Mengunduh/mengekspor `yolo11n.pt` resmi Ultralytics (COCO 80 kelas) ke format ONNX (`models/model1.onnx`).

---

### 2. C++ Inference Engine Updates

#### [MODIFY] [main.cpp](file:///Ubuntu/home/baim/workspace/yolo-project/inference/src/main.cpp)
- **Update Class Mapping**:
  - Sesuaikan `namespace Model1` dengan indeks standar COCO:
    - `PERSON = 0`
    - `CAR = 2`
    - `MOTORCYCLE = 3`
    - `BUS = 5`
    - `TRUCK = 7`
- **Update Inisialisasi Model 1**:
  - Konfigurasi `num_classes = 80` beserta daftar 80 nama kelas COCO.
- **Perbaiki Visualisasi**:
  - Gambar box dan label teks untuk semua objek terdeteksi (`car`, `bus`, `truck`, `person`, `motorcycle`).
  - Motor tetap diproses ke Stage 2 (`processMotorcycle`) untuk plat + OCR + helm (`model2`).
- **Perbaiki Output Directory**:
  - Simpan hasil gambar ke `../data/output/` (atau fallback sesuai struktur direktori input/output project).

---

## Verification Plan

### Automated / Command-Line Verification
1. Export model ONNX baru:
   ```bash
   python3 -c "from ultralytics import YOLO; YOLO('yolo11n.pt').export(format='onnx', imgsz=640)"
   mv yolo11n.onnx models/model1.onnx
   ```
2. Recompile project C++:
   ```bash
   cd inference/build && cmake .. && make -j$(nproc)
   ```
3. Uji coba inferensi pada gambar input `2.jpg`:
   ```bash
   cd inference && ./build/yolo ../data/input/images/2.jpg
   ```
4. Verifikasi bahwa:
   - Mobil (Mazda & bus/mobil lain), orang (2 orang), dan plat/elemen lainnya terdeteksi serta divisualisasikan dengan benar.
   - Output tersimpan rapi di direktori `../data/output/`.
