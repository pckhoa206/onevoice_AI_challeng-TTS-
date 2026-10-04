# 📑 BÁO CÁO TÓM TẮT DÀNH CHO LEADER & HỘI ĐỒNG GIÁM KHẢO
## ONEVOICE AI — QUALCOMM HEXAGON NPU TTS DEPLOYMENT (SUPERTONIC 3)
### THIẾT BỊ MỤC TIÊU: QUALCOMM DRAGONWING IQ-9075 EVK & SAMSUNG GALAXY S24 ULTRA (SNAPDRAGON 8 GEN 3)

---

## 🎯 1. TỔNG QUAN DỰ ÁN TRONG 3 ĐIỂM CỐT LÕI

1. **Mục tiêu kỹ thuật:** Chuyển đổi toàn diện hệ thống Text-to-Speech đa ngữ thế hệ mới (**Supertonic 3**, cơ chế Flow-Matching Continuous Normalizing Flow ODE) sang kiến trúc **Pure NPU (Lượng hóa hỗn hợp W8A16)**, chạy trực tiếp trên bộ xử lý thần kinh **Qualcomm Hexagon HTP NPU Core**.
2. **Ngôn ngữ trọng tâm:** **Tiếng Anh (English - `en`)** và **Tiếng Hàn (Korean - `ko`)**, mở rộng hỗ trợ Tiếng Việt (`vi`) và Tiếng Trung (`zh`).
3. **Phần cứng mục tiêu thực chứng:**
   - **Qualcomm Dragonwing IQ-9075 EVK:** SoC QCS9075, Hexagon HTP v73 NPU, Qualcomm Linux 1.9 (Nền tảng công nghiệp/Robotics).
   - **Samsung Galaxy S24 Ultra:** SoC Snapdragon 8 Gen 3, Hexagon HTP v75 NPU (Nền tảng thiết bị di động thương mại).

---

## ⚖️ 2. PHÂN ĐỊNH PHẦN CỨNG TRUNG THỰC: CPU HOST VS HEXAGON NPU

Theo triết lý chuẩn công nghiệp **Đồng xử lý không đối xứng (Asymmetric Co-Processing)**:

| Tác vụ / Submodel | Phần cứng đảm nhiệm | Tỷ trọng FLOPs | Độ trễ (Latency) | Trạng thái kỹ thuật |
| :--- | :---: | :---: | :---: | :--- |
| **QNN Tokenizer (C++ QnnOpPackage)** | **CPU Host / DSP HVX** | $< 0.01\%$ | **$4.0\text{ – }4.6\text{ }\mu\text{s}$** | C++ flat table lookup + bit-shift Hangul Jamo, nhanh gấp **$4.8\times$** Python. Không tạo đồ thị ONNX do HTP thiếu phép `Mod` và dynamic string expansion. |
| **1. Duration Predictor (W8A16)** | **Hexagon HTP NPU** | $\sim 2.0\%$ | **$4.9\text{ ms}$** (S24)<br>**$5.4\text{ ms}$** (Dragonwing) | In-Graph Dynamic Mask + Fused Speed-Scaling. Zero CPU compute. |
| **2. Text Encoder (W8A16)** | **Hexagon HTP NPU** | $\sim 8.0\%$ | **$2.8\text{ ms}$** (S24)<br>**$7.2\text{ ms}$** (Dragonwing) | One-Hot GEMM ($O(1)$ memory access, tận dụng Systolic Array NPU) + In-Graph Dynamic Mask. |
| **3. Vector Estimator (W8A16)** | **Hexagon HTP NPU** (Single-Step)<br>+ Host điều phối vòng lặp ODE | $\sim 15.0\%$ | **$72.3\text{ ms}$ / bước**<br>Peak RAM: **$15.15\text{ MB}$** | In-Graph Static Noise Buffer (tiết kiệm 57.6 KB DMA). Khử nhiễu Flow ODE trên NPU silicon. |
| **4. Neural Vocoder (W8A16)** | **Hexagon HTP NPU** | **$\sim 75.0\text{ – }85.0\%$** | **$7.4\text{ ms}$** (QNN Binary)<br>**$43.3\text{ ms}$** (Dragonwing) | **Offload 100% sang NPU SRAM (0% CPU Fallback)**. Fused FIR 16kHz + PCM16 trực tiếp trong đồ thị. **RTF < 0.0016** (nhanh gấp **>625 lần** thời gian thực). |

---

## 🚀 3. NĂM ĐỘT PHÁ TỐI ƯU HÓA ĐỒ THỊ CHUYÊN SÂU (TASKS 5 – 9)

1. **Tác vụ 5 (One-Hot GEMM):** Thay thế toán tử `Gather` tra cứu 8,322 ký tự bằng `OneHot + MatMul`. Biến việc truy xuất bộ nhớ rời rạc (gây Cache Miss và stall NPU) thành phép nhân ma trận song song trực tiếp trên mảng tâm thu (Systolic Array). **Bảo toàn 100% độ chính xác số học (Cosine Sim = 1.000000)**.
2. **Tác vụ 6 (In-Graph Dynamic Mask):** Nhúng bộ sinh mặt nạ nhị phân vào trực tiếp đồ thị ONNX thông qua chuỗi toán tử `Sub -> Clip -> Less -> Cast`. Triệt tiêu hoàn toàn việc truyền các mảng `text_mask`, `latent_mask` qua lại giữa Host và NPU.
3. **Tác vụ 7 (In-Graph Static Noise Buffer):** Nhúng mốc nhiễu chuẩn định $x_0 \sim \mathcal{N}(0, I)$ dạng tĩnh vào SRAM/Initializer của NPU. Loại bỏ việc CPU phải gọi `np.random.randn()` và triệt tiêu 57.6 KB truyền DMA qua bus FastRPC.
4. **Tác vụ 8 (Fused Polyphase FIR 16kHz + PCM16 Vocoder):** Tích hợp bộ lọc thông thấp 63-tap Hamming Low-Pass và ép kiểu số nguyên INT16 ngay tại tầng xuất của Vocoder. Giảm **63.7%** dung lượng payload âm thanh chuyển về Host, triệt tiêu 100% thời gian chạy `resample_poly` trên CPU.
5. **Tác vụ 9 (Qualcomm QNN C++ Custom Tokenizer):** Xây dựng theo chuẩn Qualcomm QNN OpPackage specification (`libQnnSupertonicTokenizer`), phân rã ký tự Hangul bằng phép dịch bit toán học ($SIndex = cp - 0xAC00$), bảng tra cứu tĩnh 128 KB `.rodata`. Đạt độ trễ **$4.3\text{ }\mu\text{s}$**, sẵn sàng tích hợp thẳng vào Hexagon DSP/HVX.

---

## 📊 4. KẾT QUẢ BENCHMARK PHẦN CỨNG THỰC TẾ TRÊN QUALCOMM AI HUB

### 4.1. Samsung Galaxy S24 Ultra (Snapdragon 8 Gen 3 - Hexagon HTP v75)
* **Job Quantize W8A16:** DP ([`j5689q6vg`](https://workbench.aihub.qualcomm.com/jobs/j5689q6vg/)), TE ([`jpyokn475`](https://workbench.aihub.qualcomm.com/jobs/jpyokn475/)), Vocoder ([`jgjr6qv7p`](https://workbench.aihub.qualcomm.com/jobs/jgjr6qv7p/)).
* **Độ trễ NPU (Hardware Latency):** DP **$4.9\text{ ms}$** | TE **$2.8\text{ ms}$** | Vocoder **$34.3\text{ ms}$** (QNN Binary: **$7.4\text{ ms}$**).
* **Âm thanh thực tế từ NPU S24:**
  - Tiếng Anh ([`live_tasks_567_english.wav`](../outputs/aihub_live_spoken_speech/live_tasks_567_english.wav) - Job [`jgk2w8o2g`](https://workbench.aihub.qualcomm.com/jobs/jgk2w8o2g/)): Voice Band Energy **`97.53%`**, Trọng tâm phổ **$2,332.7\text{ Hz}$**.
  - Tiếng Hàn ([`live_tasks_567_korean.wav`](../outputs/aihub_live_spoken_speech/live_tasks_567_korean.wav) - Job [`jp2rqev4g`](https://workbench.aihub.qualcomm.com/jobs/jp2rqev4g/)): Voice Band Energy **`96.16%`**, Trọng tâm phổ **$3,874.9\text{ Hz}$**.

### 4.2. Qualcomm Dragonwing IQ-9075 EVK (SoC QCS9075 - Hexagon HTP v73)
* **Job Quantize W8A16:** DP ([`jp1nllyng`](https://workbench.aihub.qualcomm.com/jobs/jp1nllyng/)), TE ([`jgdd99e6g`](https://workbench.aihub.qualcomm.com/jobs/jgdd99e6g/)), VE ([`jp4yook2p`](https://workbench.aihub.qualcomm.com/jobs/jp4yook2p/)), Vocoder ([`jpxljjn8p`](https://workbench.aihub.qualcomm.com/jobs/jpxljjn8p/)).
* **Độ trễ NPU (Hardware Latency):** DP **$5.4\text{ ms}$** (RAM 19.3 MB) | TE **$7.2\text{ ms}$** (RAM 19.8 MB) | VE **$72.3\text{ ms}$** (RAM 15.2 MB) | Vocoder **$43.3\text{ ms}$** (RAM 62.6 MB).
* **Kỹ thuật tách câu dài (Chunked Assembly):** Cắt câu dài thành các mệnh đề $< 64$ token, suy luận trên phần cứng NPU Dragonwing, ghép nối với nhịp nghỉ tự nhiên $120\text{ ms}$:
  - Tiếng Anh ([`live_dragonwing_fresh_english_complete.wav`](../outputs/aihub_live_spoken_speech/live_dragonwing_fresh_english_complete.wav)): **$12.42\text{ s}$**, Voice Band Energy **`78.02%`**, Trọng tâm phổ **$2,362.9\text{ Hz}$**, Đánh giá **VERIFIED_AUDIBLE_SPEECH**.
  - Tiếng Hàn ([`live_dragonwing_fresh_korean_complete.wav`](../outputs/aihub_live_spoken_speech/live_dragonwing_fresh_korean_complete.wav)): **$12.20\text{ s}$**, Voice Band Energy **`79.01%`**, Trọng tâm phổ **$2,475.0\text{ Hz}$**, Đánh giá **VERIFIED_AUDIBLE_SPEECH**.
* **Báo cáo JSON nghiệm thu:** [`live_dragonwing_fresh_quantized_report.json`](../outputs/aihub_live_spoken_speech/live_dragonwing_fresh_quantized_report.json).

---

## 🎧 5. VỊ TRÍ TÀI NGUYÊN NGHIỆM THU (KEY ASSETS)

- **Mã nguồn lõi C++ Tokenizer:** [`src/step3_tts/qnn_custom_tokenizer/`](../src/step3_tts/qnn_custom_tokenizer/)
- **Mã nguồn TTS Engine Pure NPU:** [`src/step3_tts/supertonic_pure_npu_v2_engine.py`](../src/step3_tts/supertonic_pure_npu_v2_engine.py)
- **File âm thanh sinh từ NPU Dragonwing:** [`outputs/aihub_live_spoken_speech/`](../outputs/aihub_live_spoken_speech/)
- **Báo cáo đo đạc chuyên sâu:**
  - [02_hexagon_npu_deployment_report.md](02_hexagon_npu_deployment_report.md)
  - [04_full_npu_deployment_and_tradeoff_report.md](04_full_npu_deployment_and_tradeoff_report.md)
  - [report_cpu.md](report_cpu.md)
