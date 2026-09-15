# Workflow CI Bank Marketing

Jonathan Christian Herutomo — Kriteria 3 submission MSML Dicoding.

Workflow melatih ulang LogisticRegression melalui MLflow Project, membuat image dengan `mlflow models build-docker`, menguji endpoint prediksi, lalu mengirim image ke Docker Hub. Model juga disimpan sebagai artefak GitHub Actions dan lampiran release repository.

Model menggunakan konfigurasi baseline Kriteria 2: `max_iter=2000` dan `random_state=42`. Average precision validasi baseline 0.203243 lebih tinggi daripada kandidat boosting terbaik 0.155238. Model ini digunakan untuk demonstrasi MLOps; recall positif validasi pada ambang 0.5 masih nol. Keberhasilan CI tidak membuktikan kelayakan model untuk penggunaan bisnis.

## Menjalankan lokal

Gunakan Python 3.12.7. Dari folder repository:

```powershell
python -m pip install -r MLProject/requirements.txt
mlflow run MLProject --env-manager local --experiment-name Bank-Marketing-CI
```

`conda.yaml` juga disertakan untuk eksekusi dengan Conda. Opsi `--env-manager local` memakai environment aktif; workflow memasang dependensi yang sama sebelum menjalankan project. Folder keluaran `MLProject/outputs` harus baru atau kosong untuk setiap run. Untuk menyimpan run lokal berikutnya secara terpisah, gunakan parameter seperti `-P output_dir=outputs-run2`.

Data train dan validasi merupakan salinan hasil preprocessing Kriteria 1. CI tidak membaca data test. Model menerima 62 fitur numerik dengan nama dan urutan sesuai signature; jangan memberikan data mentah atau melakukan scaling ulang. Data berasal dari [UCI Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank), berlisensi CC BY 4.0. Metadata preprocessing mencatat sumber dan keterbatasan evaluasinya.

## GitHub Actions dan Docker Hub

Workflow terpicu oleh perubahan file project/workflow pada `main`, atau tombol **Run workflow**. Konfigurasi repository yang diperlukan:

- Actions variable `DOCKERHUB_USERNAME`: `jojookun`.
- Actions secret `DOCKERHUB_TOKEN`: personal access token Docker Hub dengan izin Read & Write.

Token diisi langsung melalui Settings GitHub, tidak disimpan dalam kode. Workflow hanya berjalan pada push ke `main` atau pemicu manual; pull request tidak diberi akses token publikasi. Izin `contents: write` dipakai untuk membuat release dengan token sementara GitHub Actions.

Setiap build menjalankan container sementara dan membandingkan sepuluh prediksi endpoint `/invocations` dengan model hasil training. Image dipublikasikan hanya jika pengujian berhasil. Tag `run-<run_id>-<run_attempt>` menyimpan versi masing-masing run; tag `latest` diperbarui pada publikasi yang sukses. Digest image dicatat di lampiran release.

Artefak Actions disimpan selama 90 hari. Release repository menyimpan `bank-marketing-model.tar.gz` berisi model MLflow, informasi run, contoh request, prediksi pembanding, bukti pengujian serving, dan digest image. File arsip ini merupakan keluaran CI; jangan memasukkannya sebagai ZIP bersarang dalam paket submission akhir.

Dengan Docker yang sudah berjalan, image dapat dilayani secara lokal:

```powershell
docker pull jojookun/bank-marketing-msml:latest
docker run --rm -p 127.0.0.1:5001:8080 jojookun/bank-marketing-msml:latest
```

Untuk reproduksi, gunakan tag run atau digest yang tercatat pada release. Jalankan `python verify_serving.py` dari repository setelah artefak run yang sesuai tersedia di `MLProject/outputs`.

Referensi: [MLflow Projects](https://mlflow.org/docs/latest/ml/projects), [MLflow build-docker](https://mlflow.org/docs/2.22.0/api_reference/cli.html), [Docker login action](https://github.com/docker/login-action).
