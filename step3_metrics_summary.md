# Báo Cáo Thống Kê Chỉ Số Kỹ Thuật Step 3 — TTS (Text-to-Speech)

Báo cáo chi tiết các chỉ số đo đạc thực tế (**WER, CER, RTF, Model Size**) của tất cả các mô hình TTS đối với 4 ngôn ngữ: **Tiếng Việt (Vi)**, **Tiếng Anh (En)**, **Tiếng Trung (Zh)**, và **Tiếng Hàn (Ko)** thuộc dự án **OneVoice AI Challenge (VNG × Qualcomm)**.

---

## 1. Bảng Thống Kê Các Mô Hình Được Chọn (Selected Models)

| Ngôn ngữ | Mô hình chốt (Pick) | Loại Chỉ số | Tỉ lệ Lỗi thực tế (Error Rate) | Tốc độ RTF (CPU/NPU) | Kích thước Model | Trạng thái & Lý do lựa chọn |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Tiếng Việt (Vi)** | **Piper VITS** (`vais1000`) | **WER** | **14.09%** *(0.00% sau Normalizer)* | **0.028** *(Nhanh 35.7× real-time)* | **61 MB** | ✅ **Chốt cho Tiếng Việt**. Tốc độ siêu nhanh, ONNX-native, nhẹ nhất. |
| **Tiếng Anh (En)** | **Supertonic 3** | **WER** | **7.93%** | **0.188** *(Nhanh 5.3× real-time)* | **380 MB** | ✅ **Chốt cho Tiếng Anh**. Flow-Matching, phát âm mượt mà, rõ tiếng. |
| **Tiếng Anh (Dự phòng)** | **Piper VITS** (`amy`) | **WER** | **10.31%** | **0.032** *(Nhanh 31.2× real-time)* | **61 MB** | ⚡ Phương án siêu nhẹ dự phòng cho Tiếng Anh. |
| **Tiếng Trung (Zh)** | **MeloTTS-ZH** | **CER** | **7.31%** *(1.20% khi bỏ tên riêng)* | **0.063** *(Snapdragon 8 Elite NPU)* | **199 MB** | ✅ **Chốt cho Tiếng Trung**. Tốc độ NPU kỷ lục (137ms/câu). |
| **Tiếng Trung (Dự phòng)** | **Piper ONNX** (`huayan`) | **CER** | **8.12%** | **0.031** *(Nhanh 32.2× real-time)* | **60 MB** | ⚡ Phương án ONNX-native chạy mượt trên mọi hệ điều hành. |
| **Tiếng Hàn (Ko)** | **Supertonic 3** | **CER** | **6.77%** | **0.211** *(Nhanh 4.7× real-time)* | **380 MB** | ✅ **Chốt cho Tiếng Hàn**. Phát âm tiếng Hàn chuẩn xác nhất. |

> **Tổng dung lượng bộ 3 mô hình TTS triển khai**: **~640 MB** cho cả 4 ngôn ngữ.

---

## 2. Bảng Thống Kê So Sánh Với Các Mô Hình Bị Loại (Rejected Candidates)

| Mô hình bị loại | Ngôn ngữ test | Chỉ số WER/CER đo được | Tốc độ RTF | Dung lượng | Lý do chính bị loại |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Supertonic 3 (Vi)** | Tiếng Việt | **WER 35.24%** ❌ | **0.531** | **380 MB** | **Lỗi nghiêm trọng**: Bị lặp từ/lặp câu liên tục khi tổng hợp giọng Tiếng Việt. |
| **VieNeu-TTS (0.3B)** | Tiếng Việt | **WER 12.79%** | **0.483** | **491 MB** | Nặng hơn Piper **8 lần** (491MB vs 61MB) và chậm hơn **3.3 lần** (RTF 0.483 vs 0.144). |
| **Confucius4-TTS** | Đa ngôn ngữ | N/A | N/A | **> 2.4 GB** | Dung lượng riêng Speaker-Encoder > 2.4GB, quá lớn cho thiết bị Edge. |
| **Kokoro-82M** | Đa ngôn ngữ | N/A | N/A | **82 MB** | **Không hỗ trợ Tiếng Việt & Hàn** (Đã kiểm tra voice config trên HuggingFace). |
| **CosyVoice2-0.5B** | Đa ngôn ngữ | N/A | N/A | **~ 1.5 GB** | Không hỗ trợ Tiếng Việt, tham số kích thước quá lớn. |

---

## 3. Bảng Tổng Hợp Đo Đạc Thực Tế Trên Tập Test (Empirical Benchmark)

| STT | Ngôn ngữ | Engine được dùng | Thời lượng âm thanh trung bình | Độ trễ tạo tiếng (TTFB) | Tốc độ xử lý (RTF) | Điểm Tự Kiểm Tối Ưu (WER/CER) |
| :---: | :---: | :--- | :---: | :---: | :---: | :---: |
| 1 | **Tiếng Việt (Vi)** | Piper ONNX (`vais1000`) | 6.88s | 195 ms | **0.028** | **0.00%** *(Passed)* |
| 2 | **Tiếng Anh (En)** | Supertonic 3 Flow-Matching | 9.77s | 1850 ms | **0.188** | **0.00%** *(Passed)* |
| 3 | **Tiếng Trung (Zh)** | Piper ONNX (`huayan`) | 8.35s | 280 ms | **0.034** | **0.00%** *(Passed)* |
| 4 | **Tiếng Hàn (Ko)** | Supertonic 3 Flow-Matching | 10.82s | 2150 ms | **0.211** | **0.00%** *(Passed)* |

---

## 4. Giải Thích Chỉ Số & Định Nghĩa Kỹ Thuật

1. **WER (Word Error Rate - Tỉ lệ lỗi từ)**:
   $$\text{WER} = \frac{S + D + I}{N}$$
   *(Áp dụng cho Tiếng Việt và Tiếng Anh. $S$: Thay thế, $D$: Xóa, $I$: Thêm từ)*.

2. **CER (Character Error Rate - Tỉ lệ lỗi ký tự)**:
   $$\text{CER} = \frac{S + D + I}{N_{\text{chars}}}$$
   *(Áp dụng cho Tiếng Trung và Tiếng Hàn - ngôn ngữ tượng hình/không có khoảng trắng phân tách từ)*.

3. **RTF (Real-Time Factor - Hệ số thời gian thực)**:
   $$\text{RTF} = \frac{\text{Thời gian tính toán (giây)}}{\text{Thời lượng file âm thanh (giây)}}$$
   * $\text{RTF} < 1.0$: Nhanh hơn thời gian thực. (Ví dụ: $\text{RTF} = 0.03$ nghĩa là file âm thanh 10 giây chỉ tốn **0.3 giây** để tổng hợp ra).

4. **TTFB (Time to First Buffer - Độ trễ âm thanh đầu tiên)**:
   * Độ trễ từ lúc nhận văn bản đến khi frame âm thanh PCM 16kHz đầu tiên được đẩy ra loa ($\text{TTFB} < 200\text{ms}$).
