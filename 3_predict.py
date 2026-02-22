import os
import numpy as np
import librosa
import pickle
import warnings

warnings.filterwarnings('ignore')

MODEL_DIR = "models"
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
SVM_MODEL_PATH = os.path.join(MODEL_DIR, "svm_model.pkl")
KNN_MODEL_PATH = os.path.join(MODEL_DIR, "knn_model.pkl")
TARGET_SR = 16000


def extract_mfcc(file_path):
    try:
        y, sr = librosa.load(file_path, sr=TARGET_SR)

        #chia thành 0.25s đẻ có 4000 mẫu cho MFCC
        if len(y) < sr * 0.25:
            return None

        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        delta_mfccs = librosa.feature.delta(mfccs)
        delta2_mfccs = librosa.feature.delta(mfccs, order=2)

        combined_mfccs = np.vstack((mfccs, delta_mfccs, delta2_mfccs))

        mfccs_mean = np.mean(combined_mfccs.T, axis=0)
        mfccs_std = np.std(combined_mfccs.T, axis=0)

        return np.hstack((mfccs_mean, mfccs_std))
    except Exception:
        return None


def predict_with_both_models(audio_path):
    if not all(os.path.exists(p) for p in [SCALER_PATH, SVM_MODEL_PATH, KNN_MODEL_PATH]):
        print("[-] Lỗi: Thiếu mô hình hoặc file Scaler.")
        return

    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    with open(SVM_MODEL_PATH, 'rb') as f:
        svm_model = pickle.load(f)
    with open(KNN_MODEL_PATH, 'rb') as f:
        knn_model = pickle.load(f)

    try:
        print(f"\n[*] Đang phân tích file: {audio_path} ...")

        raw_features = extract_mfcc(audio_path)
        if raw_features is None:
            print("[-] Lỗi: File quá ngắn hoặc hỏng.")
            return

        features_scaled = scaler.transform(raw_features.reshape(1, -1))

        svm_pred = svm_model.predict(features_scaled)[0]
        svm_prob = np.max(svm_model.predict_proba(features_scaled)[0]) * 100

        knn_pred = knn_model.predict(features_scaled)[0]
        knn_prob = np.max(knn_model.predict_proba(features_scaled)[0]) * 100

        print("Kết quả dự đoán:")
        print("1. Mô hình SVM:")
        print(f"   -> Người đang nói: {svm_pred}")
        print(f"   -> Độ tin cậy:     {svm_prob:.2f}%\n")
        print("2. Mô hình KNN:")
        print(f"   -> Người đang nói: {knn_pred}")
        print(f"   -> Độ tin cậy:     {knn_prob:.2f}%")

        if svm_pred == knn_pred:
            print(f"\n[+] Kết luận: Cả hai mô hình đều đồng ý đây là giọng của: {svm_pred}")
        else:
            print(f"\n[!] Hai mô hình cho kết quả khác nhau.")

    except Exception as e:
        print(f"[-] Lỗi xử lý file: {e}")


if __name__ == "__main__":
    test_audio_file = "data/processed/recording.wav"

    if os.path.exists(test_audio_file):
        predict_with_both_models(test_audio_file)
    else:
        print(f"[-] Không tìm thấy file {test_audio_file}.")