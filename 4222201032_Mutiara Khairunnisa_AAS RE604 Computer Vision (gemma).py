import os
import csv
import base64
import requests
import numpy as np
from PIL import Image
from difflib import SequenceMatcher

# === KONFIGURASI ===
TEST_FOLDER = r"C:\Users\Toshiba\Downloads\RE604\AAS\test"
GROUND_TRUTH_FILE = os.path.join(TEST_FOLDER, "generate_ground_truth.csv")
OUTPUT_CSV = os.path.join(TEST_FOLDER, "hasil_prediksi.csv")
MODEL_NAME = "google/gemma-3-4b"
LMSTUDIO_API_URL = "http://192.168.1.8:1234"

# === FUNGSI CER ===
import numpy as np

def calculate_cer_details(reference: str, hypothesis: str):
    ref = list(reference)
    hyp = list(hypothesis)
    N = len(ref)

    # Matriks dynamic programming
    dp = np.zeros((len(ref) + 1, len(hyp) + 1), dtype=int)
    backtrace = [[None] * (len(hyp) + 1) for _ in range(len(ref) + 1)]

    for i in range(len(ref) + 1):
        dp[i][0] = i
        if i > 0:
            backtrace[i][0] = 'D'
    for j in range(len(hyp) + 1):
        dp[0][j] = j
        if j > 0:
            backtrace[0][j] = 'I'

    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
                backtrace[i][j] = 'M'  # Match
            else:
                sub = dp[i - 1][j - 1] + 1  # Substitution
                ins = dp[i][j - 1] + 1      # Insertion
                delete = dp[i - 1][j] + 1   # Deletion

                min_val = min(sub, ins, delete)
                dp[i][j] = min_val
                if min_val == sub:
                    backtrace[i][j] = 'S'
                elif min_val == ins:
                    backtrace[i][j] = 'I'
                else:
                    backtrace[i][j] = 'D'

    # Hitung jumlah S, D, I
    i, j = len(ref), len(hyp)
    S = D = I = 0
    while i > 0 or j > 0:
        action = backtrace[i][j]
        if action == 'M' or action == 'S':
            if action == 'S':
                S += 1
            i -= 1
            j -= 1
        elif action == 'I':
            I += 1
            j -= 1
        elif action == 'D':
            D += 1
            i -= 1

    CER = (S + D + I) / max(N, 1)
    return round(CER, 3), S, D, I, N


# === QUERY gemma MELALUI LMStudio ===
def query_gemma(image_path, prompt_text):
    try:
        with open(image_path, "rb") as img_file:
            img_bytes = img_file.read()
        img_base64 = base64.b64encode(img_bytes).decode("utf-8")
    except Exception as e:
        print(f"[ERROR] Gagal membuka file: {image_path} - {e}")
        return "ERROR"

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}},
                    {"type": "text", "text": prompt_text}
                ]
            }
        ],
        "temperature": 0.2
    }

    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(f"{LMSTUDIO_API_URL}/v1/chat/completions", json=payload, headers=headers, timeout=150)
        response.raise_for_status()
        data = response.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "ERROR")
        return content.strip()
    except Exception as e:
        print(f"[LMStudio ERROR] {e}")
        return "ERROR"

# === MEMBUAT GROUND TRUTH OTOMATIS ===
def create_ground_truth_file():
    image_files = [f for f in os.listdir(TEST_FOLDER) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    with open(GROUND_TRUTH_FILE, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['image_name', 'ground_truth'])
        for img in image_files:
            gt_label = os.path.splitext(img)[0]
            writer.writerow([img, gt_label])
    print(f"[INFO] Ground truth otomatis dibuat: {GROUND_TRUTH_FILE} ({len(image_files)} entri)")

# === LOAD GROUND TRUTH ===
def load_ground_truth():
    if not os.path.exists(GROUND_TRUTH_FILE):
        print("[INFO] File ground truth tidak ditemukan. Membuat otomatis...")
        create_ground_truth_file()

    gt_dict = {}
    with open(GROUND_TRUTH_FILE, newline='') as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            if len(row) >= 2:
                gt_dict[row[0].strip()] = row[1].strip()
            else:
                print(f"[WARNING] Baris tidak valid di CSV: {row}")
    return gt_dict

# === PEMROSESAN HASIL PREDIKSI ===
def clean_prediction(pred):
    pred = ''.join(c for c in pred if c.isalnum() or c == ' ')
    return pred.upper().strip()

# === EVALUASI OCR ===
def run_ocr_evaluation():
    ground_truth = load_ground_truth()
    image_files = [f for f in os.listdir(TEST_FOLDER) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    results = []

    for img_name in sorted(image_files):
        img_path = os.path.join(TEST_FOLDER, img_name)
        prompt = "What is the license plate number shown in this image? Respond only with the plate number."
        prediction = query_gemma(img_path, prompt)

        prediction_cleaned = clean_prediction(prediction)
        gt = ground_truth.get(img_name, "").upper()

        if prediction != "ERROR":
            score, S, D, I, N = calculate_cer_details(gt, prediction_cleaned)
        else:
            score, S, D, I, N = 1.0, 0, 0, 0, len(gt)

        print(f"[{img_name}] GT: {gt} | Prediksi: {prediction_cleaned} | CER: {score}")
        results.append([img_name, gt, prediction_cleaned, score])

    with open(OUTPUT_CSV, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['image_name', 'ground_truth', 'prediction', 'CER_score'])
        writer.writerows(results)

    print(f"[SELESAI] Hasil evaluasi disimpan di: {OUTPUT_CSV}")


# === JALANKAN ===
if __name__ == "__main__":
    
    run_ocr_evaluation()
