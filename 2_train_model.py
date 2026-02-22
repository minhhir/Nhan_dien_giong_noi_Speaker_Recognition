import os
import numpy as np
import librosa
import pickle
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

PROCESSED_DATA_DIR = "data/processed"
MODEL_DIR = "models"
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


def load_data():
    X, y = [], []
    print("[*] Đang trích xuất đặc trưng MFCC siêu cấp (240 chiều)...")

    for person_name in os.listdir(PROCESSED_DATA_DIR):
        person_dir = os.path.join(PROCESSED_DATA_DIR, person_name)
        if not os.path.isdir(person_dir):
            continue

        files = [f for f in os.listdir(person_dir) if f.endswith(".wav")]
        for file_name in tqdm(files, desc=f"Xử lý {person_name}", unit="file"):
            file_path = os.path.join(person_dir, file_name)
            features = extract_mfcc(file_path)

            if features is not None:
                X.append(features)
                y.append(person_name)

    return np.array(X), np.array(y)


if __name__ == "__main__":
    X, y = load_data()
    print(f"\n[+] Dữ liệu sẵn sàng: {X.shape[0]} mẫu âm thanh (240 đặc trưng).")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\n Training SVM...")
    svm_model = SVC(kernel='linear', probability=True)
    svm_model.fit(X_train_scaled, y_train)
    svm_acc = accuracy_score(y_test, svm_model.predict(X_test_scaled))
    print(f"Độ chính xác SVM: {svm_acc * 100:.2f}%")

    print("Training KNN...")
    knn_model = KNeighborsClassifier(n_neighbors=3)
    knn_model.fit(X_train_scaled, y_train)
    knn_acc = accuracy_score(y_test, knn_model.predict(X_test_scaled))
    print(f"Độ chính xác KNN: {knn_acc * 100:.2f}%")

    os.makedirs(MODEL_DIR, exist_ok=True)

    with open(os.path.join(MODEL_DIR, "scaler.pkl"), 'wb') as f:
        pickle.dump(scaler, f)
    with open(os.path.join(MODEL_DIR, "svm_model.pkl"), 'wb') as f:
        pickle.dump(svm_model, f)
    with open(os.path.join(MODEL_DIR, "knn_model.pkl"), 'wb') as f:
        pickle.dump(knn_model, f)

    print(f"\n[+] Đã lưu Scaler, SVM và KNN tại thư mục: {MODEL_DIR}")