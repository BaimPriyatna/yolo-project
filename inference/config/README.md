# Class list untuk Model 1

`main.cpp` membaca daftar nama class dari `../models/model1_classes.txt` (relatif dari folder `build/`)
saat runtime — **bukan** hardcode di kode C++. Ini supaya ganti model (COCO pretrained <-> custom hasil
training) cukup ganti isi file itu, tanpa perlu edit/recompile `main.cpp`.

## Cara pakai

Copy salah satu template di folder ini ke `models/model1_classes.txt`, sesuaikan sama `model1.onnx`
yang sedang dipakai:

- **`coco80_classes.txt`** — kalau `model1.onnx` masih COCO pretrained (yolo11n dsb, 80 class)
- **`custom3_classes.txt`** — kalau `model1.onnx` hasil training custom (car, motorcycle, person)

```bash
cp inference/config/coco80_classes.txt models/model1_classes.txt
# atau
cp inference/config/custom3_classes.txt models/model1_classes.txt
```

**Penting:** urutan baris di file ini HARUS PERSIS SAMA dengan urutan class output model (urutan
`names:` di `data.yaml` waktu training, atau urutan standar COCO kalau pakai pretrained). Nama
`"person"`, `"motorcycle"`, `"car"` wajib ada persis (case-sensitive) karena `main.cpp` mencari
class ini berdasarkan nama. `"bus"` dan `"truck"` opsional — kalau model-nya nggak punya 2 class ini
(mis. model custom 3-class), fitur deteksi bus/truck otomatis nonaktif, sisanya tetap jalan normal.
