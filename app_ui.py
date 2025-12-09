import sys
import cv2
import pickle
import numpy as np
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QPushButton, 
                             QFileDialog, QVBoxLayout, QWidget, QTableWidget, 
                             QTableWidgetItem, QHeaderView, QMessageBox)
from PyQt5.QtGui import QPixmap, QImage, QFont
from PyQt5.QtCore import Qt

class AksaraApp(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Konfigurasi Awal
        self.IMG_SIZE = (64, 64)
        self.database_pola = {}
        self.load_database()
        
        self.initUI()

    def load_database(self):
        # Cek apakah file database ada
        if os.path.exists('database_pola.pkl'):
            with open('database_pola.pkl', 'rb') as f:
                self.database_pola = pickle.load(f)
        else:
            QMessageBox.critical(self, "Error", "File 'database_pola.pkl' belum ada!\nSilakan jalankan train.py dulu.")

    def initUI(self):
        self.setWindowTitle("Deteksi Aksara Jawa (Metode Pola)")
        self.setGeometry(100, 100, 800, 600) # Ukuran jendela x, y, width, height

        # --- Widget Utama ---
        
        # 1. Label Judul
        self.label_judul = QLabel("Upload Gambar Aksara Jawa", self)
        self.label_judul.setAlignment(Qt.AlignCenter)
        self.label_judul.setFont(QFont('Arial', 14, QFont.Bold))

        # 2. Area Tampil Gambar
        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setText("Gambar akan muncul di sini")
        self.image_label.setStyleSheet("border: 2px dashed gray; background-color: #f0f0f0;")
        self.image_label.setFixedSize(300, 300)

        # 3. Tombol Pilih Gambar
        self.btn_load = QPushButton("Pilih Gambar Tes", self)
        self.btn_load.setFont(QFont('Arial', 10))
        self.btn_load.clicked.connect(self.browse_image)

        # 4. Label Hasil Prediksi Utama
        self.label_hasil = QLabel("Hasil: -", self)
        self.label_hasil.setFont(QFont('Arial', 16, QFont.Bold))
        self.label_hasil.setAlignment(Qt.AlignCenter)
        self.label_hasil.setStyleSheet("color: blue; margin-top: 10px;")

        # 5. Tabel Detail Kemiripan
        self.table_result = QTableWidget()
        self.table_result.setColumnCount(2)
        self.table_result.setHorizontalHeaderLabels(["Huruf", "Tingkat Error (Semakin kecil = Mirip)"])
        self.table_result.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        # --- Layout (Tata Letak) ---
        layout = QVBoxLayout()
        layout.addWidget(self.label_judul)
        layout.addWidget(self.btn_load)
        
        # Layout tengah (Gambar)
        layout.addWidget(self.image_label, alignment=Qt.AlignCenter)
        layout.addWidget(self.label_hasil)
        
        # Label kecil untuk tabel
        lbl_detail = QLabel("Detail Analisa Kemiripan:")
        layout.addWidget(lbl_detail)
        layout.addWidget(self.table_result)

        # Set Layout ke Window
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def browse_image(self):
        # Buka dialog pilih file
        fname, _ = QFileDialog.getOpenFileName(self, 'Pilih Gambar', '.', 'Image files (*.jpg *.png *.jpeg)')
        
        if fname:
            self.process_and_predict(fname)

    def process_and_predict(self, file_path):
        # 1. Tampilkan Gambar di UI
        pixmap = QPixmap(file_path)
        self.image_label.setPixmap(pixmap.scaled(300, 300, Qt.KeepAspectRatio))

        # 2. Baca & Proses Gambar dengan OpenCV
        img = cv2.imread(file_path)
        if img is None: 
            return

        # --- PREPROCESSING (Wajib sama dengan train.py) ---
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, img_thresh = cv2.threshold(img_gray, 150, 255, cv2.THRESH_BINARY_INV)
        img_input = cv2.resize(img_thresh, self.IMG_SIZE)
        # --------------------------------------------------

        # 3. Hitung Kemiripan dengan Database
        hasil_analisa = [] # List untuk menyimpan (nama_huruf, nilai_error)

        skor_terbaik = float('inf')
        huruf_terdeteksi = "Tidak Diketahui"

        for huruf, pola_ref in self.database_pola.items():
            # Hitung selisih pixel (Mean Squared Error sederhana)
            # absdiff menghitung beda warna pixel. Jika sama persis hasilnya 0.
            selisih = cv2.absdiff(img_input, pola_ref)
            skor_error = np.mean(selisih)
            
            hasil_analisa.append((huruf, skor_error))

            # Update jika menemukan error yang lebih kecil
            if skor_error < skor_terbaik:
                skor_terbaik = skor_error
                huruf_terdeteksi = huruf

        # 4. Update UI Hasil
        self.label_hasil.setText(f"Prediksi: Aksara '{huruf_terdeteksi.upper()}'")

        # 5. Isi Tabel Detail (Diurutkan dari error terkecil)
        hasil_analisa.sort(key=lambda x: x[1]) # Sort by skor error

        self.table_result.setRowCount(len(hasil_analisa))
        for row, (huruf, skor) in enumerate(hasil_analisa):
            self.table_result.setItem(row, 0, QTableWidgetItem(huruf))
            self.table_result.setItem(row, 1, QTableWidgetItem(f"{skor:.4f}"))

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = AksaraApp()
    ex.show()
    sys.exit(app.exec_())