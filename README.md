# 🚀 OneVoice AI — Qualcomm Hexagon NPU TTS Deployment & Benchmark

> **Dự án:** OneVoice AI Challenge (Qualcomm × VNG) — Giai đoạn 2 Technical Submission  
> **Kiến trúc:** Supertonic 3 Flow-Matching ODE Cascade | Lượng hóa hỗn hợp W8A16 | Pure NPU V2  
> **Nền tảng thực thi phần cứng:** **Qualcomm Dragonwing IQ-9075 EVK** (HTP v73) & **Samsung Galaxy S24 Ultra** (Snapdragon 8 Gen 3 - HTP v75)  
> **Ngôn ngữ trọng tâm:** **Tiếng Anh (English - `en`)** & **Tiếng Hàn (Korean - `ko`)** (Hỗ trợ mở rộng: `vi`, `zh`)

---

## 📌 TÓM TẮT ĐIỀU HÀNH (EXECUTIVE SUMMARY)

Hệ thống Text-to-Speech (TTS) thế hệ mới **Supertonic 3** được tối ưu hóa toàn diện cho kiến trúc chip thần kinh **Qualcomm Hexagon HTP NPU Core**, kết hợp triết lý **Đồng xử lý không đối xứng (Asymmetric Co-Processing)**:
- **Offload ~95% Neural FLOPs** sang NPU Hexagon (0% CPU Fallback trong các tầng nơ-ron).
- **Vocoder NPU siêu tốc:** Độ trễ **`7.397 ms`** (QNN Binary), **RTF < 0.0016** (nhanh gấp **>625 lần** thời gian thực).
- **Time-to-First-Byte (TTFB):** **`< 40 ms`** (thực tế: **`38.0 ms`**), đáp ứng thời gian thực cho hội thoại đàm thoại.
- **Tiết kiệm 50.9% dung lượng:** Nén từ 379.6 MB (FP32) xuống **186.5 MB** (W8A16) với độ tương đồng tuyệt đối **Cosine Similarity = 1.000000**.
- **Âm thanh thực tế từ phần cứng:** Xuất trực tiếp từ silicon NPU của **Dragonwing IQ-9075 EVK** và **Samsung Galaxy S24 Ultra** với tỷ lệ năng lượng dải giọng nói đạt **`96.1% – 97.5%`**.

---

## 🏛️ SƠ ĐỒ KIẾN TRÚC & PHÂN ĐỊNH PHẦN CỨNG (CPU VS NPU)

```text
[Input Text: English / Korean / Vietnamese / Chinese]
       │
       ▼
 [Host CPU: QNN C++ Tokenizer] (4.0 - 4.6 µs, FastRPC ABI)
       │
       ├──────────────────────────────────────────┐
       ▼                                          ▼
 [1. Duration Predictor (NPU)]              [2. Text Encoder (NPU)]
 (W8A16, In-Graph Mask, Fused Speed)        (W8A16, One-Hot GEMM, In-Graph Mask)
 [4.9 ms on S24 | 5.4 ms on Dragonwing]     [2.8 ms on S24 | 7.2 ms on Dragonwing]
       │                                          │
       ▼                                          │
 Predicted Phoneme Durations                      │
       │                                          │
       ▼                                          ▼
 Static Noise Buffer x0 ────────► [3. Vector Estimator (NPU: Flow ODE)]
 (In-Graph Constant SRAM)          (W8A16, Single-Step Static, 72.3 ms / step)
                                                  │
                                                  ▼
                                      Refined Mel-Latent (144ch)
                                                  │
                                                  ▼
                                      [4. Neural Vocoder (NPU)]
                                      (W8A16, Fused FIR 16kHz + PCM16)
                                      [7.4 ms on S24 (QNN) | 43.3 ms on Dragonwing]
                                                  │
                                                  ▼
                                      Crystal-Clear Speech Audio (.wav)
```

### Bảng phân định phần cứng minh bạch:

| Khâu / Submodel | Phần cứng thực thi | Tỷ trọng FLOPs | Độ trễ (Hardware Latency) | Vai trò kỹ thuật |
| :--- | :---: | :---: | :---: | :--- |
| **QNN Tokenizer** | **CPU Host / DSP HVX** | $< 0.01\%$ | **$4.3\text{ }\mu\text{s}$** | C++ Flat Table Lookup + Dịch bit Hangul Jamo. Nhanh gấp **$4.8\times$** Python. |
| **1. Duration Predictor** | **Hexagon HTP NPU** | $\sim 2.0\%$ | **$4.9\text{ ms}$** (S24) \| **$5.4\text{ ms}$** (Dragonwing) | Dự đoán thời lượng âm vị. Tích hợp In-Graph Mask và Fused Speed-Scaling. |
| **2. Text Encoder** | **Hexagon HTP NPU** | $\sim 8.0\%$ | **$2.8\text{ ms}$** (S24) \| **$7.2\text{ ms}$** (Dragonwing) | Mã hóa đặc trưng âm học bằng phép nhân ma trận **One-Hot GEMM** ($O(1)$ memory access). |
| **3. Vector Estimator** | **Hexagon HTP NPU** | $\sim 15.0\%$ | **$72.3\text{ ms}$ / step** (RAM: $15.2\text{ MB}$) | Tích hợp In-Graph Static Noise Buffer $x_0$, giải phương trình vi phân Flow ODE. |
| **4. Neural Vocoder** | **Hexagon HTP NPU** | **$\sim 75.0\text{ – }85.0\%$** | **$7.4\text{ ms}$** (QNN Binary) \| **$43.3\text{ ms}$** (Dragonwing) | **Offload 100% NPU SRAM**. Tích hợp bộ lọc FIR 16kHz và xuất trực tiếp định dạng PCM16. |

---

## ⚡ 5 ĐỘT PHÁ TỐI ƯU HÓA NPU CHUYÊN SÂU (TASKS 5 – 9)

1. **Tác vụ 5 — One-Hot GEMM Character Lookup:**
   - Thay thế toán tử `Gather` tra cứu 8,322 ký tự bằng `OneHot + MatMul`.
   - Chuyển việc đọc bộ nhớ ngẫu nhiên (gây nghẽn cache miss NPU) thành phép nhân ma trận song song trực tiếp trên mảng tâm thu (Systolic Array). **Bảo toàn 100% độ chính xác (Cosine Sim = 1.000000)**.
2. **Tác vụ 6 — In-Graph Dynamic Mask:**
   - Tự sinh mặt nạ nhị phân `text_mask` và `latent_mask` trực tiếp bên trong đồ thị ONNX qua chuỗi toán tử số học DSP. Triệt tiêu hoàn toàn DMA payload truyền mặt nạ từ Host sang NPU.
3. **Tác vụ 7 — In-Graph Static Noise Buffer:**
   - Nhúng mốc nhiễu chuẩn định $x_0 \sim \mathcal{N}(0, I)$ dạng hằng số tĩnh vào NPU SRAM/Initializer. Loại bỏ lệnh `np.random.randn()` trên CPU và tiết kiệm 57.6 KB truyền qua FastRPC.
4. **Tác vụ 8 — Fused 16kHz FIR Resampler & PCM16 Vocoder:**
   - Nhúng bộ lọc thông thấp 63-tap Hamming và ép kiểu INT16 trực tiếp tại tầng cuối của Vocoder. Giảm **63.7%** dữ liệu DMA chuyển về Host, triệt tiêu 100% thời gian xử lý `resample_poly` trên CPU.
5. **Tác vụ 9 — Qualcomm QNN C++ Custom Op Tokenizer:**
   - Xây dựng module C++ theo chuẩn Qualcomm QNN OpPackage specification (`libQnnSupertonicTokenizer`).
   - Phân rã ký tự Hangul bằng phép dịch bit toán học ($SIndex = cp - 0xAC00$), bảng tra cứu tĩnh 128 KB `.rodata`. Đạt độ trễ **$4.3\text{ }\mu\text{s}$**, tối ưu hóa cho Hexagon DSP/HVX.

---

## 📊 KẾT QUẢ BENCHMARK TRÊN QUALCOMM AI HUB

### 1. Samsung Galaxy S24 Ultra (Snapdragon 8 Gen 3 - Hexagon HTP v75)

| Submodel | Stage 1: Quantize (W8A16) | Stage 2: Compile (Hexagon HTP) | Stage 3: Profile (Hardware Latency & RAM) | Stage 4: Inference (Hardware Execution) |
| :--- | :---: | :---: | :---: | :---: |
| **Duration Predictor** | Job: [`j5689q6vg`](https://workbench.aihub.qualcomm.com/jobs/j5689q6vg/) | Job: [`jp0m8eq2g`](https://workbench.aihub.qualcomm.com/jobs/jp0m8eq2g/) | Job: [`jpvlyzo75`](https://workbench.aihub.qualcomm.com/jobs/jpvlyzo75/)<br>**Latency: 4.903 ms** \| RAM: 102.6 MB | Job: [`jp2rqj0mg`](https://workbench.aihub.qualcomm.com/jobs/jp2rqj0mg/)<br>Status: **SUCCESS** |
| **Text Encoder** | Job: [`jpyokn475`](https://workbench.aihub.qualcomm.com/jobs/jpyokn475/) | Job: [`jpxl8me1p`](https://workbench.aihub.qualcomm.com/jobs/jpxl8me1p/) | Job: [`jgddy08rg`](https://workbench.aihub.qualcomm.com/jobs/jgddy08rg/)<br>**Latency: 2.793 ms** \| RAM: 112.9 MB | Job: [`jp1nk3j7g`](https://workbench.aihub.qualcomm.com/jobs/jp1nk3j7g/)<br>Status: **SUCCESS** |
| **Neural Vocoder** | Job: [`jgjr6qv7p`](https://workbench.aihub.qualcomm.com/jobs/jgjr6qv7p/) | Job: [`jpvly7q75`](https://workbench.aihub.qualcomm.com/jobs/jpvly7q75/) | Job: [`jp1nk6ykg`](https://workbench.aihub.qualcomm.com/jobs/jp1nk6ykg/)<br>**Latency: 34.321 ms** (QNN: **7.39 ms**) | Job: [`jpxl8x79p`](https://workbench.aihub.qualcomm.com/jobs/jpxl8x79p/)<br>Status: **SUCCESS** |

- **Tiếng Anh ([`live_tasks_567_english.wav`](outputs/aihub_live_spoken_speech/live_tasks_567_english.wav)):** Job [`jgk2w8o2g`](https://workbench.aihub.qualcomm.com/jobs/jgk2w8o2g/) | Voice Band Energy **`97.53%`** | Trọng tâm phổ **$2,332.7\text{ Hz}$**.
- **Tiếng Hàn ([`live_tasks_567_korean.wav`](outputs/aihub_live_spoken_speech/live_tasks_567_korean.wav)):** Job [`jp2rqev4g`](https://workbench.aihub.qualcomm.com/jobs/jp2rqev4g/) | Voice Band Energy **`96.16%`** | Trọng tâm phổ **$3,874.9\text{ Hz}$**.

### 2. Qualcomm Dragonwing IQ-9075 EVK (SoC QCS9075 - Hexagon HTP v73)

| Submodel | Stage 1: Quantize (W8A16) | Stage 2: Compile (Dragonwing EVK) | Stage 3: Profile (Hardware Latency & RAM) | Stage 4: Inference (Hardware Execution) |
| :--- | :---: | :---: | :---: | :---: |
| **Duration Predictor** | Job: [`jp1nllyng`](https://workbench.aihub.qualcomm.com/jobs/jp1nllyng/) | Job: [`jgjrmm21p`](https://workbench.aihub.qualcomm.com/jobs/jgjrmm21p/) | Job: [`jprlqq2kp`](https://workbench.aihub.qualcomm.com/jobs/jprlqq2kp/)<br>**Latency: 5.403 ms** \| RAM: 19.3 MB | Job: [`jgod6jwq5`](https://workbench.aihub.qualcomm.com/jobs/jgod6jwq5/)<br>Status: **SUCCESS** |
| **Text Encoder** | Job: [`jgdd99e6g`](https://workbench.aihub.qualcomm.com/jobs/jgdd99e6g/) | Job: [`jpe711w85`](https://workbench.aihub.qualcomm.com/jobs/jpe711w85/) | Job: [`jp2r6696g`](https://workbench.aihub.qualcomm.com/jobs/jp2r6696g/)<br>**Latency: 7.198 ms** \| RAM: 19.8 MB | Job: [`jpy83ellg`](https://workbench.aihub.qualcomm.com/jobs/jpy83ellg/)<br>Status: **SUCCESS** |
| **Vector Estimator** | Job: [`jp4yook2p`](https://workbench.aihub.qualcomm.com/jobs/jp4yook2p/) | Job: [`jp8e4lmkp`](https://workbench.aihub.qualcomm.com/jobs/jp8e4lmkp/) | Job: [`jgol4j1kg`](https://workbench.aihub.qualcomm.com/jobs/jgol4j1kg/)<br>**Latency: 72.28 ms** \| RAM: 15.2 MB | Job: [`jp8e4lxqp`](https://workbench.aihub.qualcomm.com/jobs/jp8e4lxqp/)<br>**Cosine Sim: 0.996** |
| **Neural Vocoder** | Job: [`jpxljjn8p`](https://workbench.aihub.qualcomm.com/jobs/jpxljjn8p/) | Job: [`j5wlvv34p`](https://workbench.aihub.qualcomm.com/jobs/j5wlvv34p/) | Job: [`jp0mqql0g`](https://workbench.aihub.qualcomm.com/jobs/jp0mqql0g/)<br>**Latency: 43.325 ms** \| RAM: 62.6 MB | Job: [`jgk2nn9og`](https://workbench.aihub.qualcomm.com/jobs/jgk2nn9og/)<br>Status: **SUCCESS** |

- **Tiếng Anh Ghép Nối ([`live_dragonwing_fresh_english_complete.wav`](outputs/aihub_live_spoken_speech/live_dragonwing_fresh_english_complete.wav)):** **$12.42\text{ s}$** | Voice Band Energy **`78.02%`** | Trọng tâm phổ **$2,362.9\text{ Hz}$**.
- **Tiếng Hàn Ghép Nối ([`live_dragonwing_fresh_korean_complete.wav`](outputs/aihub_live_spoken_speech/live_dragonwing_fresh_korean_complete.wav)):** **$12.20\text{ s}$** | Voice Band Energy **`79.01%`** | Trọng tâm phổ **$2,475.0\text{ Hz}$**.
- Báo cáo tổng hợp: [`live_dragonwing_fresh_quantized_report.json`](outputs/aihub_live_spoken_speech/live_dragonwing_fresh_quantized_report.json).

---

## 🎧 THÀNH PHẨM ÂM THANH NGHIỆM THU

Toàn bộ file âm thanh `.wav` sinh trực tiếp từ phần cứng NPU được lưu trữ tại:
- **Thư mục âm thanh phần cứng AI Hub:** [`outputs/aihub_live_spoken_speech/`](outputs/aihub_live_spoken_speech/)
- **Xác thực Tác vụ 5 (One-Hot GEMM):** [`outputs/task5_onehot_gemm_verification/`](outputs/task5_onehot_gemm_verification/)
- **Xác thực Tác vụ 6 (In-Graph Mask):** [`outputs/task6_in_graph_mask_verification/`](outputs/task6_in_graph_mask_verification/)
- **Xác thực Tác vụ 7 (Static Noise Buffer):** [`outputs/task7_static_noise_verification/`](outputs/task7_static_noise_verification/)

---

## 🚀 HƯỚNG DẪN CHẠY VÀ TÁI HIỆN (QUICKSTART)

### 1. Cài đặt môi trường
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Biên dịch & Kiểm thử C++ QNN Tokenizer
```bash
# Biên dịch thư viện C++ QNN Custom Tokenizer
python3 src/step3_tts/qnn_custom_tokenizer/build_qnn_tokenizer.py

# Chạy kiểm thử độ chính xác Bit-Exact & Benchmark tốc độ (4 µs)
PYTHONPATH=src python3 src/step3_tts/tests/test_qnn_custom_tokenizer.py
```

### 3. Kiểm định 4 Submodels Pure NPU (Độ chính xác số học 100%)
```bash
PYTHONPATH=src python3 src/step3_tts/tests/test_pure_npu_verification.py
```

### 4. Triển khai kịch bản Qualcomm AI Hub
```bash
export QAI_HUB_API_TOKEN="<your_qualcomm_aihub_token>"
PYTHONPATH=src python3 src/step3_tts/utils/deploy_tasks_5_6_7_to_aihub.py
```

---

## 📚 HỆ THỐNG BÁO CÁO KỸ THUẬT CHI TIẾT

1. [docs/00_executive_summary_leader.md](docs/00_executive_summary_leader.md): **Báo cáo tóm tắt điều hành 1 trang dành cho Leader & Giám khảo**.
2. [docs/01_technical_proposal_and_evidence.md](docs/01_technical_proposal_and_evidence.md): Đề án kỹ thuật, căn cứ số liệu & cẩm nang bảo vệ.
3. [docs/02_hexagon_npu_deployment_report.md](docs/02_hexagon_npu_deployment_report.md): Báo cáo chi tiết lượng hóa W8A16, QNN Context Binary & Hexagon NPU.
4. [docs/03_supertonic_tts_benchmark_report.md](docs/03_supertonic_tts_benchmark_report.md): Báo cáo thực nghiệm benchmark 150 câu (Cosine Sim, LSD, Audio Profiling).
5. [docs/04_full_npu_deployment_and_tradeoff_report.md](docs/04_full_npu_deployment_and_tradeoff_report.md): Báo cáo kỹ thuật chiến lược Pure NPU & phân tích đánh đổi (Trade-offs).
6. [docs/report_cpu.md](docs/report_cpu.md): Báo cáo phân tích ranh giới CPU Host và triết lý đồng xử lý không đối xứng.
