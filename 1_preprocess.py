import librosa
import soundfile as sf
import os
from tqdm import tqdm

RAW_DATA_DIR = "data/raw"
PROCESSED_DATA_DIR = "data/processed"
CHUNK_DURATION = 1.0  # cắt thành các đoạn 1 giây
TARGET_SR = 16000 # chuẩn hóa tần số mẫu về 16kHz

def process_person_folder(person_name):
    person_raw_dir = os.path.join(RAW_DATA_DIR, person_name)
    person_processed_dir = os.path.join(PROCESSED_DATA_DIR, person_name)
    os.makedirs(person_processed_dir, exist_ok=True)

    chunk_count = 1
    print(f"\n[*] Đang xử lý dữ liệu của: {person_name}")

    files = [f for f in os.listdir(person_raw_dir) if f.endswith(".wav")]

    for file_name in tqdm(files, desc="Đang cắt file", unit="file"):
        file_path = os.path.join(person_raw_dir, file_name)
        try:
            y, sr = librosa.load(file_path, sr=TARGET_SR)
            y_trimmed, _ = librosa.effects.trim(y, top_db=30)
            chunk_samples = int(CHUNK_DURATION * sr)
            total_samples = len(y_trimmed)

            for start_i in range(0, total_samples, chunk_samples):
                end_i = start_i + chunk_samples
                chunk = y_trimmed[start_i:end_i]

                if len(chunk) == chunk_samples:
                    output_path = os.path.join(person_processed_dir, f"{chunk_count}.wav")
                    sf.write(output_path, chunk, sr)
                    chunk_count += 1

        except Exception as e:
            print(f"[-] Lỗi khi xử lý file {file_name}: {e}")

    print(f"  -> Đã tạo thành công {chunk_count - 1} mẫu âm thanh (1 giây/mẫu)")

if __name__ == "__main__":
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

    for item in os.listdir(RAW_DATA_DIR):
        person_dir = os.path.join(RAW_DATA_DIR, item)
        if os.path.isdir(person_dir):
            process_person_folder(item)

    print("\n[+] Xong! Đã chuẩn bị xong dữ liệu huấn luyện.")