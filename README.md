# 🎙️ Hệ thống Nhận dạng Người nói (Speaker Recognition)

Dự án ứng dụng Học máy (Machine Learning) để phân tích, trích xuất đặc trưng sinh trắc học từ giọng nói và xác thực danh tính người dùng. 
* **Thực hiện bởi:** Sinh viên lớp CNTT 65 nào đó - Trường Đại học Kinh tế Quốc dân (NEU).
* **Công cụ chính:** Python, Librosa, Scikit-Learn,pickle( và các thư viện python khác).

---

## 📖 I. Cơ sở lý thuyết về Xử lý Âm thanh (Audio Processing)
*(Tham khảo kiến thức cốt lõi từ các bài toán nhận dạng âm thanh)*

Âm thanh tự nhiên là sóng liên tục (Analog). Để máy tính hiểu được, chúng ta phải chuyển nó thành dữ liệu rời rạc (Digital) thông qua các bước tối ưu sau:

### 1. Lấy mẫu (Sampling)
* **Lý thuyết:** Quá trình "chụp ảnh" lại biên độ sóng âm tại các khoảng thời gian đều nhau. Số lần chụp trong 1 giây gọi là Tần số lấy mẫu (Sampling Rate).
* **Ứng dụng trong dự án:** Hệ thống ép toàn bộ file ghi âm (từ các điện thoại/mic khác nhau) về chuẩn **16,000 Hz (16kHz)**. Đây là tần số chuẩn quốc tế để bắt trọn vẹn dải âm của giọng nói con người mà không làm máy tính bị quá tải do dữ liệu thừa.

### 2. Lọc khoảng lặng (Voice Activity Detection - VAD)
* **Lý thuyết:** Giọng nói thường chứa rất nhiều khoảng im lặng (khi ngập ngừng, thở). Đưa các đoạn im lặng này vào huấn luyện sẽ làm nhiễu mô hình.
* **Ứng dụng:** Code sử dụng hàm `librosa.effects.trim` với ngưỡng `top_db=30` để tự động dò và gọt bỏ các khoảng lặng ở hai đầu file, đảm bảo 100% dữ liệu đi vào mô hình là dữ liệu "có tiếng người".

---

## 🧬 II. Trích xuất đặc trưng Giọng nói (Feature Extraction)
Máy tính không thể học trực tiếp từ biểu đồ sóng âm (Waveform). Chúng ta phải dịch giọng nói thành các con số toán học đại diện cho "cấu tạo sinh lý" của vòm họng người nói. Dự án sử dụng bộ 3 đặc trưng siêu việt:

### 1. MFCC (Mel-Frequency Cepstral Coefficients)

* **Lý thuyết:** Là kỹ thuật mô phỏng lại cấu trúc của ốc tai con người. Thính giác người nhạy cảm với âm trầm và kém nhạy với âm cao (thang đo Mel). MFCC nắm bắt được **đặc điểm hình học của thanh quản, vòm họng và lưỡi** tại thời điểm phát âm.
* **Ứng dụng:** Trích xuất 40 hệ số MFCC cơ bản đại diện cho "Khẩu hình".

### 2. Delta và Delta-Delta MFCC (Đạo hàm bậc 1 và bậc 2)

* **Lý thuyết:** Nếu chỉ dùng MFCC, máy tính chỉ thấy được một "bức ảnh tĩnh" của khẩu hình. Nhưng khi nói chuyện (đặc biệt là giọng hài hước, luyến láy), khẩu hình thay đổi liên tục. 
    * **Delta (Vận tốc):** Tính tốc độ thay đổi khẩu hình từ âm tiết này sang âm tiết khác.
    * **Delta-Delta (Gia tốc):** Tính độ giật, độ rung của dây thanh quản khi người nói nhấn giọng.
* **Ứng dụng:** Kết hợp MFCC (40) + Delta (40) + Delta-Delta (40) tạo thành một ma trận **120 chiều**. 
* **Tối ưu hóa (Vector hóa):** Để đưa vào mô hình học máy, hệ thống tính toán giá trị Trung bình (Mean) và Độ lệch chuẩn (Standard Deviation) theo thời gian của 120 chiều này $\rightarrow$ Tạo ra vector đại diện cuối cùng gồm **240 đặc trưng** cực kỳ mạnh mẽ cho mỗi file âm thanh.

---

## 🤖 III. Các mô hình Máy học (Machine Learning Models)
Trong dự án này, hệ thống so sánh trực tiếp 2 thuật toán phân loại kinh điển: **K-Nearest Neighbors (KNN)** và **Support Vector Machine (SVM)**. Đặc trưng đầu vào là vector âm thanh không gian nhiều chiều (240 chiều). 

⚠️ **Lưu ý tiền xử lý (Rất quan trọng):** Cả KNN và SVM đều là các thuật toán cực kỳ nhạy cảm với độ lớn của dữ liệu (Distance/Margin-based). Do đó, bộ dữ liệu bắt buộc phải đi qua bước **Chuẩn hóa (StandardScaler)** để đưa mọi đặc trưng về phân phối chuẩn (Mean = 0, Variance = 1) trước khi đưa vào huấn luyện.

### 1. K-Nearest Neighbors (KNN) - Thuật toán Láng giềng gần nhất
KNN là một trong những thuật toán học máy có giám sát (Supervised Learning) đơn giản nhưng hiệu quả nhất, thuộc nhóm **Instance-based learning** (Học dựa trên cá thể) và **Lazy Learning** (Học lười biếng).

* **Cơ chế hoạt động:** * "Học lười biếng" nghĩa là KNN gần như không có pha huấn luyện (Training phase) rõ ràng. Nó đơn giản là ghi nhớ toàn bộ tập dữ liệu âm thanh.
  * Ở pha dự đoán (Prediction phase), khi có một giọng nói mới, KNN sẽ tính toán khoảng cách (thường dùng khoảng cách **Euclidean** hoặc **Manhattan**) từ điểm dữ liệu mới này đến tất cả các điểm trong không gian 240 chiều của tập huấn luyện.
  * Sau đó, thuật toán lọc ra **K điểm gần nhất** (K-Nearest Neighbors) và sử dụng cơ chế **Bầu chọn đa số (Majority Voting)** để quyết định danh tính người nói.
* **Tham số K (`n_neighbors=3`):** * Việc chọn K lẻ (K=3) giúp tránh hiện tượng hòa (tie) khi bầu chọn.
  * Nếu K quá nhỏ (K=1): Mô hình dễ bị nhiễu (Overfitting).
  * Nếu K quá lớn: Ranh giới phân loại mờ nhạt (Underfitting), dễ bị chệch hướng bởi các nhãn chiếm đa số.
* **Đánh giá trong dự án:** KNN rất dễ cài đặt, tuy nhiên khi dữ liệu âm thanh lớn dần, tốc độ nhận dạng (Predict) của KNN sẽ chậm đi đáng kể vì nó phải tính khoảng cách với mọi mẫu trong kho dữ liệu gốc.

### 2. Support Vector Machine (SVM) - Máy học Vector hỗ trợ
Trái ngược với sự đơn giản của KNN, SVM là một mô hình toán học chặt chẽ và mạnh mẽ hơn rất nhiều, đặc biệt tối ưu với các không gian dữ liệu nhiều chiều (High-dimensional space) như mảng 240 đặc trưng MFCC của chúng ta.

* **Cơ chế hoạt động (Hyperplane & Margin):** * Thay vì đi tìm "hàng xóm", mục tiêu của SVM là tìm ra một **Siêu mặt phẳng (Hyperplane)** tốt nhất để chia cắt không gian dữ liệu thành các vùng riêng biệt đại diện cho từng người nói.
  * Siêu mặt phẳng tốt nhất là mặt phẳng tạo ra được **Lề (Margin) lớn nhất**. Margin chính là khoảng cách từ siêu mặt phẳng đến các điểm dữ liệu gần nhất của mỗi lớp.
  * Những điểm dữ liệu nằm sát ngay trên đường biên giới này được gọi là các **Support Vectors** (Vector hỗ trợ). Chúng là những điểm nòng cốt duy nhất quyết định vị trí của mặt phẳng, các điểm dữ liệu nằm sâu bên trong vương quốc của mình sẽ không làm ảnh hưởng đến mô hình.
* **Kernel Trick (Thủ thuật hạt nhân):** * Giọng nói là dữ liệu phi tuyến tính (Non-linear). Đôi khi ta không thể dùng một mặt phẳng thẳng để chia cắt giọng của A và B. 
  * Nhờ hàm **Kernel**, SVM có thể "phóng" dữ liệu từ không gian ban đầu lên một chiều không gian cao hơn rất nhiều. Ở không gian mới này, dữ liệu bỗng nhiên có thể phân tách tuyến tính dễ dàng.
  * Trong dự án này, sau khi trích xuất ra tận 240 đặc trưng, không gian gốc đã đủ phức tạp, nên ta dùng `kernel='linear'` (Kernel tuyến tính) để phân chia tối ưu hóa tốc độ.
* **Đánh giá trong dự án:** SVM cực kỳ hiệu quả. Dù người dùng có giả giọng hay giọng bị méo (Vector bị trượt đi), miễn là nó chưa vượt rào qua "Margin" đã được thiết lập chặt chẽ của SVM, mô hình vẫn nhận ra danh tính một cách cực kỳ ổn định.

## 🚀 IV. Hướng dẫn sử dụng
### Cấu trúc dự án
```text
Speaker_Recognition/
│
├── data/
│   ├── raw/                 # Chứa thư mục con tên từng người (file .wav gốc dài)
│   └── processed/           # Chứa các file tự động cắt nhỏ 1 giây
│
├── models/                  # Lưu trữ scaler.pkl, svm_model.pkl, knn_model.pkl
│
├── 1_preprocess.py          # Script 1: Tiền xử lý (Cắt 1s, Lọc khoảng lặng)
├── 2_train_model.py         # Script 2: Trích xuất 240 MFCC & Train mô hình
├── 3_predict.py             # Script 3: Xác thực giọng nói thực tế
└── requirements.txt         # Thư viện cần thiết
```
⚙️ Hướng dẫn chạy
1. Cài đặt thư viện: ```pip install -r requirements.txt```
2. Chạy file số 1 để cắt âm thanh: `python 1_preprocess.py`
3. Chạy file số 2 để huấn luyện mô hình: `python 2_train_model.py`
4. Chạy file số 3 để kiểm thử: `python 3_predict.py` 
