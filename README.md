# Deteksi_Aksara_Jawa
Sistem Deteksi Citra Aksara Jawa Berdasarkan Analisis Kemiripan Pixel (Pixel Difference Analysis)

Nama : Silvina Hariati 

NRP : 225210734


# DESKRIPSI SISTEM
Aksara Jawa (Hanacaraka) merupakan warisan budaya yang memiliki struktur visual yang unik. Dalam bidang Computer Vision, pengenalan karakter biasanya menggunakan metode kompleks seperti Convolutional Neural Network (CNN). Namun, untuk kebutuhan pembelajaran dasar dan sistem dengan komputasi ringan, diperlukan pendekatan alternatif yang lebih sederhana tanpa melibatkan arsitektur Deep Learning. Penelitian ini berfokus pada metode Pixel Difference Analysis (Analisis Perbedaan Pixel) untuk mencocokkan pola input dengan database referensi.

1.	Mengimplementasikan algoritma pengolahan citra digital dasar antara lain : Proses Grayscale (merubah RGB menjadi BW), Thresholding, Resizing.

2.	Membangun antarmuka pengguna (GUI) menggunakan PyQt5 untuk memvisualisasikan proses deteksi dan tingkat kemiripan data.


# METODOLOGI DAN PERSIAPAN
Sistem dibangun menggunakan bahasa pemrograman Python. Sebelum memulai pengembangan, pustaka (library) pendukung diinstal melalui terminal dengan perintah berikut:

codeBash
pip install PyQt5 opencv-python numpy


# Sumber Data (Dataset)
Penelitian ini menggunakan dua jenis sumber data:

1.	Data Training (Referensi): Dataset digital huruf Aksara Jawa dasar yang diperoleh dari repositori publik Kaggle. Data ini digunakan sebagai "cetakan" atau pola baku yang bersih.

2.	Data Testing (Pengujian): Citra tulisan tangan asli yang ditulis di atas kertas putih biasa menggunakan spidol/pulpen, kemudian difoto menggunakan kamera Handphone. Hal ini bertujuan untuk menguji ketahanan sistem terhadap input dunia nyata.


# PERANCANGAN DAN IMPLEMENTASI SISTEM

1.	Preprocessing (Pra-pemrosesan):
Setiap gambar (baik training maupun testing) harus melalui tahapan standarisasi agar bisa dibandingkan ("Apple to Apple").

o	Grayscale: Mengubah citra berwarna (RGB) menjadi citra kelabu (1 channel) untuk menghilangkan bias warna.

o	Thresholding (Binarisasi): Ini adalah tahap krusial untuk menemukan "data unik" atau bentuk tegas huruf. Pixel diubah menjadi hitam mutlak (0) atau putih mutlak (255).

	Tujuan: Memisahkan objek tulisan (foreground) dari latar belakang kertas (background) dan menghilangkan noise bayangan dari foto kamera HP.

o	Resizing: Mengubah ukuran citra menjadi dimensi tetap (64x64 pixel) agar matriks dapat dioperasikan secara matematika.


2.	Pembentukan Model Referensi (train.py):
Sistem membaca folder dataset Kaggle, mengambil satu sampel terbaik untuk setiap huruf (Ha, Na, Ca, dst), memprosesnya, dan menyimpannya sebagai dictionary pola dalam file database_pola.pkl.

3.	Proses Matching / Prediksi (app_ui.py):
Saat pengguna mengunggah gambar tes:

o	Gambar tes diproses (Grayscale -> Threshold -> Resize).

o	Sistem melakukan operasi pengurangan matriks absolut (Absolute Difference) antara gambar tes dengan setiap gambar di database.

o	Rumus: Error = Rata-rata (|Pixel_Input - Pixel_Database|)

