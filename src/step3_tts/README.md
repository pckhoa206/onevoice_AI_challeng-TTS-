# 🎙️ STEP 3: TEXT-TO-SPEECH (TTS) — SUPERTONIC 3 W8A16 & PURE QUALCOMM NPU

Module Text-to-Speech (TTS) trong dự án **OneVoice AI** hỗ trợ 4 ngôn ngữ (Tiếng Việt, Tiếng Anh, Tiếng Trung, Tiếng Hàn) được tối ưu hóa lượng hóa hỗn hợp **W8A16**, tái cấu trúc đồ thị sâu (**Graph Refactoring**) và thực thi trên **Qualcomm Hexagon HTP NPU (0% CPU Fallback trong tính toán nơ-ron)**.

---

## 📑 TÀI LIỆU CHI TIẾT

Toàn bộ báo cáo kỹ thuật, cơ chế lượng hóa, kết quả đo đạc phần cứng và cẩm nang bảo vệ đề án được lưu trữ trong thư mục `docs/`:

1. [01_technical_proposal_and_evidence.md](../../docs/01_technical_proposal_and_evidence.md): Đề án kỹ thuật, Căn cứ số liệu & Cẩm nang bảo vệ.
2. [02_hexagon_npu_deployment_report.md](../../docs/02_hexagon_npu_deployment_report.md): Báo cáo kỹ thuật quy trình Deploy, Tối ưu hóa đồ thị & Thực thi trên Qualcomm Hexagon NPU.
3. [03_supertonic_tts_benchmark_report.md](../../docs/03_supertonic_tts_benchmark_report.md): Báo cáo thực nghiệm 150 câu benchmark mở rộng & Độ chính xác số học.
4. [report_cpu.md](../../docs/report_cpu.md): Ranh giới CPU Host vs Hexagon NPU.

---

## 🚀 CẤU TRÚC CODEBASE STEP 3

```text
src/step3_tts/
├── supertonic_pure_npu_v2_engine.py   # Engine chính chạy 4 submodel NPU đã refactor
├── supertonic_w8a16_engine.py          # Engine lượng hóa W8A16 kiểm thử cục bộ
├── text_normalizer.py                  # Bộ chuẩn hóa chuỗi và số cho 4 ngôn ngữ
├── style_prompt_manager.py             # Quản lý voice style vector (M1/F1)
├── prosody_enhancer.py                 # Phân tích ngữ điệu và ngắt nghỉ âm
├── tests/
│   ├── test_pure_npu_verification.py   # Kiểm tra Cosine Similarity với FP32
│   └── test_pure_npu_v2_accuracy.py    # Kiểm thử độ chính xác số học
└── utils/
    ├── refactor_pure_npu_v2.py         # Script tự động sửa lỗi đồ thị cho NPU
    ├── deploy_dragonwing_iq9075_pipeline.py # Pipeline chạy AI Hub Live Hardware
    └── compile_pure_npu_w8a16.py       # Biên dịch QNN Context Binary W8A16
```

---

## ⚡ HƯỚNG DẪN THỰC THI NHANH

```bash
# 1. Chạy kiểm thử độ chính xác số học của 4 submodel NPU:
python src/step3_tts/tests/test_pure_npu_verification.py

# 2. Chạy tổng hợp giọng nói đa ngữ qua engine Pure NPU V2:
python src/step3_tts/supertonic_pure_npu_v2_engine.py
```
