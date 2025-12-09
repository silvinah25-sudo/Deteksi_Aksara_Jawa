import cv2
import os
import pickle
import numpy as np

# KONFIGURASI
DATA_SOURCE = 'dataset_aksara'
IMG_SIZE = (64, 64) # Ukuran baku

database_pola = {}

print("--- Mulai Membuat Database Pola ---")

if not os.path.exists(DATA_SOURCE):
    print(f"Error: Folder '{DATA_SOURCE}' tidak ditemukan!")
    exit()

list_huruf = sorted(os.listdir(DATA_SOURCE))

for huruf in list_huruf:
    folder_path = os.path.join(DATA_SOURCE, huruf)
    
    if os.path.isdir(folder_path):
        files = os.listdir(folder_path)
        if len(files) > 0:
            # Ambil 1 gambar pertama sebagai referensi pola utama
            path_gambar = os.path.join(folder_path, files[0])
            
            # PROSES GAMBAR (Preprocessing)
            img = cv2.imread(path_gambar)
            if img is not None:
                # 1. Grayscale
                img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                # 2. Threshold (Hitam Putih Tegas)
                _, img_thresh = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY_INV)
                # 3. Resize
                img_final = cv2.resize(img_thresh, IMG_SIZE)
                
                # Simpan ke memori
                database_pola[huruf] = img_final
                print(f"✅ Pola tersimpan: {huruf}")

# Simpan ke file
with open('database_pola.pkl', 'wb') as f:
    pickle.dump(database_pola, f)

print("--- Selesai! File 'database_pola.pkl' siap digunakan. ---")