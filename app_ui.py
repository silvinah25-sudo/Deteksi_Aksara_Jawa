import sys
import cv2
import pickle
import numpy as np
import os
import datetime
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QPushButton, 
                             QFileDialog, QVBoxLayout, QHBoxLayout, QWidget, 
                             QTableWidget, QTableWidgetItem, QHeaderView, QMessageBox, QFrame)
from PyQt5.QtGui import QPixmap, QImage, QFont
from PyQt5.QtCore import Qt

class AksaraApp(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Konfigurasi Path Laporan
        self.BASE_DIR = "laporan_output"
        self.IMG_DIR = os.path.join(self.BASE_DIR, "images")
        
        # Buat folder jika belum ada
        os.makedirs(self.IMG_DIR, exist_ok=True)
        
        # Konfigurasi Model
        self.IMG_SIZE = (64, 64)
        self.database_pola = {}
        self.load_database()
        
        self.initUI()

    def load_database(self):
        if os.path.exists('database_pola.pkl'):
            with open('database_pola.pkl', 'rb') as f:
                self.database_pola = pickle.load(f)
        else:
            QMessageBox.critical(self, "Error", "File 'database_pola.pkl' tidak ditemukan!\nJalankan train.py terlebih dahulu.")

    def initUI(self):
        self.setWindowTitle("Sistem Deteksi Aksara Jawa - Pixel Analysis")
        self.setGeometry(50, 50, 1000, 750) 
        self.setStyleSheet("background-color: #f8f9fa;")

        # --- FONT ---
        font_title = QFont('Segoe UI', 16, QFont.Bold)
        font_label = QFont('Segoe UI', 10)
        font_bold = QFont('Segoe UI', 12, QFont.Bold)

        # --- WIDGETS ---
        
        # 1. Header
        self.label_judul = QLabel("Analisa Citra Digital: Deteksi Aksara Jawa", self)
        self.label_judul.setAlignment(Qt.AlignCenter)
        self.label_judul.setFont(font_title)
        self.label_judul.setStyleSheet("color: #333; margin: 10px;")

        # 2. Area Gambar (Ada 3: Asli, Gray, Threshold)
        self.lbl_img_asli = self.create_image_box("1. Citra Asli (RGB)")
        self.lbl_img_gray = self.create_image_box("2. Grayscale")
        self.lbl_img_thresh = self.create_image_box("3. Threshold (Biner)")

        # Layout Gambar Horizontal
        layout_imgs = QHBoxLayout()
        layout_imgs.addWidget(self.lbl_img_asli)
        layout_imgs.addWidget(self.lbl_img_gray)
        layout_imgs.addWidget(self.lbl_img_thresh)

        # 3. Tombol
        self.btn_load = QPushButton("📂 Pilih Gambar & Analisa", self)
        self.btn_load.setFont(font_bold)
        self.btn_load.setStyleSheet("""
            QPushButton {
                background-color: #007bff; color: white; padding: 10px; border-radius: 5px;
            }
            QPushButton:hover { background-color: #0056b3; }
        """)
        self.btn_load.clicked.connect(self.browse_image)

        self.btn_report = QPushButton("📄 Buka Laporan HTML", self)
        self.btn_report.setFont(font_label)
        self.btn_report.setStyleSheet("""
            QPushButton {
                background-color: #28a745; color: white; padding: 10px; border-radius: 5px;
            }
            QPushButton:hover { background-color: #218838; }
        """)
        self.btn_report.clicked.connect(self.open_report)
        self.btn_report.setEnabled(False) # Matikan dulu sebelum ada hasil

        # 4. Hasil Prediksi
        self.label_hasil = QLabel("Prediksi: -", self)
        self.label_hasil.setFont(QFont('Segoe UI', 18, QFont.Bold))
        self.label_hasil.setAlignment(Qt.AlignCenter)
        self.label_hasil.setStyleSheet("color: #d63384; margin-top: 15px; border: 2px solid #d63384; padding: 10px; border-radius: 8px;")

        # 5. Tabel
        self.table_result = QTableWidget()
        self.table_result.setColumnCount(3)
        self.table_result.setHorizontalHeaderLabels(["Peringkat", "Huruf", "Tingkat Error (MSE)"])
        self.table_result.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table_result.setStyleSheet("background-color: white;")

        # --- PENYUSUNAN LAYOUT UTAMA ---
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.label_judul)
        main_layout.addWidget(self.btn_load)
        
        # Container Gambar
        group_img = QWidget()
        group_img.setLayout(layout_imgs)
        main_layout.addWidget(group_img)
        
        main_layout.addWidget(self.label_hasil)
        main_layout.addWidget(QLabel("Detail Perhitungan Matriks:"))
        main_layout.addWidget(self.table_result)
        main_layout.addWidget(self.btn_report)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

    def create_image_box(self, text):
        # Helper untuk membuat kotak gambar dengan label
        widget = QWidget()
        vbox = QVBoxLayout()
        
        lbl_text = QLabel(text)
        lbl_text.setAlignment(Qt.AlignCenter)
        lbl_text.setFont(QFont('Segoe UI', 9, QFont.Bold))
        
        lbl_img = QLabel("Kosong")
        lbl_img.setAlignment(Qt.AlignCenter)
        lbl_img.setStyleSheet("border: 1px dashed gray; background-color: #eee;")
        lbl_img.setFixedSize(200, 200)
        
        vbox.addWidget(lbl_text)
        vbox.addWidget(lbl_img)
        widget.setLayout(vbox)
        
        # Kita simpan referensi label gambar ke widget agar bisa diakses nanti
        widget.image_label = lbl_img 
        return widget

    def display_cv_image(self, cv_img, label_widget):
        # Konversi OpenCV (BGR) ke Qt (RGB) untuk ditampilkan
        if len(cv_img.shape) == 2: # Grayscale
            h, w = cv_img.shape
            bytes_per_line = w
            q_img = QImage(cv_img.data, w, h, bytes_per_line, QImage.Format_Grayscale8)
        else: # Color
            h, w, ch = cv_img.shape
            rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            bytes_per_line = ch * w
            q_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            
        pixmap = QPixmap.fromImage(q_img)
        label_widget.image_label.setPixmap(pixmap.scaled(200, 200, Qt.KeepAspectRatio))

    def browse_image(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'Pilih Gambar', '.', 'Image files (*.jpg *.png *.jpeg)')
        if fname:
            self.process_and_predict(fname)

    def process_and_predict(self, file_path):
        # 1. BACA GAMBAR
        img = cv2.imread(file_path)
        if img is None: return
        
        # Simpan gambar input asli ke folder laporan
        path_asli = os.path.join(self.IMG_DIR, "1_original.jpg")
        cv2.imwrite(path_asli, img)
        self.display_cv_image(img, self.lbl_img_asli)

        # 2. GRAYSCALE
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        path_gray = os.path.join(self.IMG_DIR, "2_grayscale.jpg")
        cv2.imwrite(path_gray, img_gray)
        self.display_cv_image(img_gray, self.lbl_img_gray)

        # 3. THRESHOLD & RESIZE
        _, img_thresh = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY_INV)
        img_input = cv2.resize(img_thresh, self.IMG_SIZE)
        
        path_thresh = os.path.join(self.IMG_DIR, "3_threshold_resized.jpg")
        cv2.imwrite(path_thresh, img_input) # Simpan gambar yang sudah di resize
        self.display_cv_image(img_thresh, self.lbl_img_thresh)

        # 4. PROSES MATCHING
        hasil_analisa = []
        skor_terbaik = float('inf')
        huruf_terdeteksi = "Tidak Diketahui"
        pola_terbaik_img = None

        for huruf, pola_ref in self.database_pola.items():
            # Hitung MSE (Mean Squared Error)
            selisih = cv2.absdiff(img_input, pola_ref)
            skor_error = np.mean(selisih)
            hasil_analisa.append((huruf, skor_error))

            if skor_error < skor_terbaik:
                skor_terbaik = skor_error
                huruf_terdeteksi = huruf
                pola_terbaik_img = pola_ref

        # Simpan Pola Database Terbaik untuk laporan
        if pola_terbaik_img is not None:
            path_ref = os.path.join(self.IMG_DIR, "4_referensi_db.jpg")
            cv2.imwrite(path_ref, pola_terbaik_img)

        # 5. UPDATE UI
        self.label_hasil.setText(f"Prediksi: Aksara '{huruf_terdeteksi.upper()}'")
        hasil_analisa.sort(key=lambda x: x[1])

        self.table_result.setRowCount(len(hasil_analisa))
        for row, (huruf, skor) in enumerate(hasil_analisa):
            self.table_result.setItem(row, 0, QTableWidgetItem(str(row + 1)))
            self.table_result.setItem(row, 1, QTableWidgetItem(huruf))
            self.table_result.setItem(row, 2, QTableWidgetItem(f"{skor:.4f}"))

        # 6. GENERATE LAPORAN HTML
        self.generate_html_report(huruf_terdeteksi, skor_terbaik, hasil_analisa)
        self.btn_report.setEnabled(True)
        QMessageBox.information(self, "Sukses", "Analisa Selesai. Laporan HTML telah dibuat.")

    def generate_html_report(self, hasil, skor, detail_list):
        # Waktu sekarang
        waktu = datetime.datetime.now().strftime("%d %B %Y, %H:%M:%S")
        
        # CSS Profesional
        css = """
        <style>
            body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f4f4f9; color: #333; margin: 0; padding: 20px; }
            .container { max-width: 900px; margin: auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 4px 10px rgba(0,0,0,0.1); }
            h1 { color: #2c3e50; text-align: center; border-bottom: 2px solid #eee; padding-bottom: 10px; }
            h2 { color: #007bff; margin-top: 30px; font-size: 1.2em; border-left: 5px solid #007bff; padding-left: 10px; }
            .meta { text-align: center; color: #777; font-size: 0.9em; margin-bottom: 20px; }
            .image-row { display: flex; justify-content: space-around; flex-wrap: wrap; margin: 20px 0; gap: 10px; }
            .image-card { text-align: center; background: #fff; padding: 10px; border: 1px solid #ddd; border-radius: 8px; width: 30%; box-shadow: 0 2px 5px rgba(0,0,0,0.05); }
            .image-card img { width: 100%; height: auto; border-radius: 4px; border: 1px solid #eee; }
            .image-card p { font-size: 0.9em; font-weight: bold; margin-top: 5px; color: #555; }
            .comparison { display: flex; align-items: center; justify-content: center; background: #e9ecef; padding: 20px; border-radius: 8px; margin: 20px 0; }
            .comparison div { text-align: center; margin: 0 20px; }
            .vs { font-weight: bold; font-size: 1.5em; color: #aaa; }
            table { width: 100%; border-collapse: collapse; margin-top: 15px; }
            th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
            th { background-color: #007bff; color: white; }
            tr:nth-child(even) { background-color: #f9f9f9; }
            tr:hover { background-color: #f1f1f1; }
            .highlight { background-color: #d4edda !important; font-weight: bold; color: #155724; }
            .footer { text-align: center; margin-top: 40px; font-size: 0.8em; color: #aaa; }
        </style>
        """

        # Generate Baris Tabel
        rows_html = ""
        for i, (h, s) in enumerate(detail_list[:10]): # Ambil top 10 saja
            cls = 'class="highlight"' if i == 0 else ''
            rows_html += f"<tr {cls}><td>{i+1}</td><td>{h}</td><td>{s:.4f}</td></tr>"

        # Konten HTML
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Laporan Deteksi Aksara Jawa</title>
            {css}
        </head>
        <body>
            <div class="container">
                <h1>Laporan Analisis Citra Digital</h1>
                <div class="meta">
                    Tanggal Analisis: {waktu} <br>
                    Sistem Deteksi Aksara Jawa (Metode: Pixel Difference)
                </div>

                <h2>1. Tahapan Preprocessing (Pra-Pemrosesan)</h2>
                <p>Berikut adalah tahapan perubahan gambar input sebelum masuk ke proses perhitungan matematika:</p>
                <div class="image-row">
                    <div class="image-card">
                        <img src="images/1_original.jpg" alt="Asli">
                        <p>1. Gambar Asli (Original)</p>
                    </div>
                    <div class="image-card">
                        <img src="images/2_grayscale.jpg" alt="Gray">
                        <p>2. Grayscale (Kelabu)</p>
                    </div>
                    <div class="image-card">
                        <img src="images/3_threshold_resized.jpg" alt="Thresh">
                        <p>3. Biner & Resize (64x64)</p>
                    </div>
                </div>
                <p style="font-size: 0.9em; color: #555;">
                    <i>Penjelasan:</i> Gambar diubah menjadi hitam putih (Thresholding) untuk memisahkan tulisan dari background kertas. Kemudian ukuran disamakan menjadi 64x64 pixel agar matriks bisa dihitung.
                </p>

                <h2>2. Proses Matching (Pencocokan)</h2>
                <p>Sistem membandingkan matriks pixel gambar input dengan database pola menggunakan metode pengurangan pixel (Subtraction).</p>
                
                <div class="comparison">
                    <div>
                        <img src="images/3_threshold_resized.jpg" width="100">
                        <p>Input (Data Tes)</p>
                    </div>
                    <div class="vs">- (Kurang) -</div>
                    <div>
                        <img src="images/4_referensi_db.jpg" width="100">
                        <p>Database (Pola {hasil})</p>
                    </div>
                    <div class="vs">=</div>
                    <div>
                        <h3>Error: {skor:.4f}</h3>
                    </div>
                </div>

                <h2>3. Hasil Perhitungan Komputasi</h2>
                <p>Tabel berikut menunjukkan urutan kemiripan dari yang paling identik (Error terendah) hingga yang paling berbeda.</p>
                <table>
                    <thead>
                        <tr>
                            <th>Peringkat</th>
                            <th>Aksara</th>
                            <th>Tingkat Error (Semakin Kecil = Mirip)</th>
                        </tr>
                    </thead>
                    <tbody>
                        {rows_html}
                    </tbody>
                </table>

                <div class="footer">
                    Generated by Python System | PyQt5 & OpenCV Implementation
                </div>
            </div>
        </body>
        </html>
        """

        # Simpan File HTML
        path_html = os.path.join(self.BASE_DIR, "laporan_deteksi.html")
        with open(path_html, "w") as f:
            f.write(html_content)
            
    def open_report(self):
        # Buka file HTML di browser default
        path_html = os.path.abspath(os.path.join(self.BASE_DIR, "laporan_deteksi.html"))
        import webbrowser
        webbrowser.open(f"file:///{path_html}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = AksaraApp()
    ex.show()
    sys.exit(app.exec_())
