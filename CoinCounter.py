import cv2
import numpy as np
import matplotlib.pyplot as plt

def deteksi_koin_lengkap(image_path):
    # --- 1. AKUISISI DATA [cite: 61, 77] ---
    img = cv2.imread(image_path)
    if img is None:
        print("Gambar tidak ditemukan.")
        return

    # Resize agar ukuran standar (Sampling) [cite: 62]
    # Menggunakan rasio aspek agar koin tidak gepeng
    target_width = 600
    h, w = img.shape[:2]
    scale = target_width / w
    dim = (target_width, int(h * scale))
    img_resized = cv2.resize(img, dim)
    
    output_img = img_resized.copy() # Untuk hasil akhir

    # --- 2. PRE-PROCESSING (RESTORASI) [cite: 63, 64] ---
    # Grayscale
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    
    # Gaussian Blur (Noise Removal)
    # Penting untuk menghilangkan detail ukiran pada koin agar tidak dianggap noise
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)

    # --- 3. SEGMENTASI (THRESHOLDING & MORFOLOGI) [cite: 65] ---
    # Adaptive Thresholding: Mengatasi pantulan cahaya pada logam koin
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)

    # Operasi Morfologi (Closing): Menutup lubang-lubang kecil pada hasil threshold
    kernel = np.ones((3,3), np.uint8)
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    # --- 4. ANALISIS BENTUK & DETEKSI ---
    contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    jumlah_koin = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0: continue

        # Rumus Circularity (Kebulatan): 1.0 = Lingkaran Sempurna
        circularity = 4 * np.pi * (area / (perimeter * perimeter))

        # Filter: Hanya ambil yang luasnya cukup besar DAN bentuknya bulat
        # Area > 500 (buang bintik debu)
        # Circularity > 0.7 (buang objek kotak/panjang)
        if area > 500 and circularity > 0.7:
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            center = (int(x), int(y))
            radius = int(radius)
            # Gambar lingkaran hijau di hasil akhir
            cv2.circle(output_img, center, radius, (0, 255, 0), 3)
            # Tulis label
            cv2.putText(output_img, "Koin", (center[0]-20, center[1]), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            jumlah_koin += 1

    # --- 5. VISUALISASI LENGKAP (SESUAI DOKUMEN)  ---
    plt.figure(figsize=(12, 10))

    # Gambar 1: Citra Asli 
    plt.subplot(2, 2, 1)
    plt.title("1. Citra Asli (Original)")
    plt.imshow(cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    # Gambar 2: Grayscale & Blur 
    plt.subplot(2, 2, 2)
    plt.title("2. Grayscale & Gaussian Blur")
    plt.imshow(blurred, cmap='gray')
    plt.axis('off')

    # Gambar 3: Segmentasi (Thresholding) 
    plt.subplot(2, 2, 3)
    plt.title("3. Segmentasi (Adaptive Threshold)")
    plt.imshow(closing, cmap='gray')
    plt.axis('off')

    # Gambar 4: Hasil Akhir 
    plt.subplot(2, 2, 4)
    plt.title(f"4. Hasil Akhir ({jumlah_koin} Koin Terdeteksi)")
    plt.imshow(cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.tight_layout()
    plt.show()

# Jalankan dengan foto koinmu
deteksi_koin_lengkap('Fle-Path')
