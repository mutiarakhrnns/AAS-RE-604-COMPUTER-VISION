# AAS-RE-604-COMPUTER-VISION

# OCR License Plate Evaluation with LMStudio & using Gemma 3 4b Models

Proyek ini merupakan skrip Python untuk **OCR (Optical Character Recognition) plat nomor kendaraan** menggunakan **Visual Language Model (VLM) Gemma** yang dijalankan melalui **LMStudio API**.  
Program akan membaca gambar dari folder tertentu, melakukan inferensi OCR, kemudian menghitung **Character Error Rate (CER)** untuk mengevaluasi performa model.

---

## Fitur Utama
- **Inferensi OCR berbasis VLM** menggunakan model `google/gemma-3-4b` via LMStudio API.
- **Pembuatan Ground Truth otomatis** dari nama file gambar.
- **Perhitungan Character Error Rate (CER)** lengkap dengan statistik Substitusi (S), Deletion (D), dan Insertion (I).
- **Penyimpanan hasil evaluasi ke CSV** untuk analisis lebih lanjut.

---
