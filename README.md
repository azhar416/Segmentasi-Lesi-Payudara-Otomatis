# Segmentasi Lesi Payudara Otomatis

Penulis:
1. Daffa Muhamad Azhar, S.Kom., M.Kom.
2. Prof. Dr. Eng. Nanik Suciati, S.Kom., M.Kom.

## Deskripsi

Program ini melakukan segmentasi lesi payudara melalui dua tahap: deteksi lesi menggunakan YOLOv9, kemudian segmentasi menggunakan MedSAM2-LoRA. Program menyimpan mask biner dengan ukuran yang sama seperti gambar masukan.

## Struktur Model

Weight model disimpan di direktori `Model/`:

- `YOLOv9_Fold1.pt`: weight YOLOv9 untuk deteksi.
- `LoRA512_R32_Fold1.pt`: weight MedSAM2-LoRA untuk segmentasi.
- `YOLOv9_Fold1.onnx`: model YOLO dalam format ONNX.

File `.pt` dilacak menggunakan Git LFS. Source SAM2 dikelola sebagai Git submodule di `program/sam2`.

## Prasyarat

- Python 3.10 atau lebih baru.
- Git dan Git LFS.
- PyTorch dan torchvision yang sesuai dengan lingkungan Python/CUDA. Program tidak memasang atau mengganti keduanya secara otomatis agar konfigurasi CUDA yang sudah tersedia tidak berubah.
- Koneksi internet pada pemanggilan pertama jika dependensi runtime lain belum tersedia.

Saat `run_inference()` dijalankan, program memeriksa dan memasang dependensi yang belum ada: OpenCV headless, Hydra, iopath, PEFT, Pillow, dan Ultralytics. Program juga menghapus `torchao` hanya jika versi yang terpasang tidak kompatibel dengan PEFT. Instalasi dependensi otomatis memerlukan akses internet.

## Mengunduh Repository

Clone repository beserta submodule:

```bash
git clone --recurse-submodules https://github.com/azhar416/Segmentasi-Lesi-Payudara-Otomatis.git
cd Segmentasi-Lesi-Payudara-Otomatis
git lfs install
git lfs pull
```

Jika repository sudah di-clone tanpa submodule, jalankan dari root repository:

```bash
git submodule update --init --recursive
git lfs pull
```

Pastikan direktori `program/sam2/sam2/` dan kedua file weight `.pt` tersedia sebelum menjalankan inferensi.

## Menjalankan dari Terminal

Jalankan perintah berikut dari root repository. Argumen pertama adalah path gambar masukan; argumen kedua adalah path file mask keluaran.

```bash
python -m program.main "path/to/input.png" "path/to/output/mask.png" --fold 1
```

Contoh di Windows PowerShell:

```powershell
python -m program.main "D:\data\gambar.png" "D:\hasil\mask.png" --fold 1
```

Argumen `--fold` menerima nilai 1 sampai 5 dan bersifat opsional. Nilai default adalah 1. Folder keluaran dibuat secara otomatis jika belum tersedia.

Untuk melihat argumen yang tersedia:

```bash
python -m program.main --help
```

## Menjalankan dari Notebook

Pastikan working directory notebook adalah root repository, kemudian panggil fungsi inferensi secara langsung:

```python
from program.main import run_inference

input_image = "/path/to/input.png"
output_image = "/path/to/output/mask.png"

mask_path = run_inference(
	image_path=input_image,
	output_mask_path=output_image,
	fold_number=1,
)
print(mask_path)
```

Pada Kaggle, gambar dan output dapat menggunakan lokasi seperti `/kaggle/input/...` dan `/kaggle/working/...`. Berikan nilai variabel path secara langsung; jangan menuliskan nama variabel sebagai string, misalnya gunakan `image_path=input_image`, bukan `image_path="input_image"`.

## Alur Inferensi

1. Gambar dibaca dan dikonversi ke RGB. Jika YOLO menemukan lebih dari satu objek, program memilih bounding box dengan confidence tertinggi.
2. YOLO menerima gambar melalui objek PIL RGB. Inferensi YOLO menggunakan confidence 0.25 dan IoU 0.45.
3. Jika tidak ada bounding box, program menggunakan point prompt tetap berdasarkan nomor fold.
4. Setelah YOLO selesai, gambar di-resize menjadi 512 x 512 untuk MedSAM2. Koordinat bounding box disesuaikan ke ukuran tersebut.
5. Mask hasil segmentasi dikembalikan ke dimensi asli gambar dan disimpan sebagai PNG biner.

## Catatan

- YOLO dijalankan pada CPU. MedSAM2 menggunakan CUDA jika tersedia, atau CPU jika tidak tersedia.
- Weight yang digunakan ditentukan oleh konstanta default di `program/main.py` dan berada di direktori `Model/`.
- Jika dependensi PyTorch atau torchvision belum tersedia, pasang build yang sesuai dengan runtime terlebih dahulu. Program sengaja tidak memilih build PyTorch secara otomatis.