# Reflection â€” Lab 22 (DPO/ORPO Alignment)

**Tên:** _<Bùi Văn Đạt>_<2A202600355>
**Cohort:** _<A20-K1>_
**Tier Ä‘Ã£ cháº¡y:** _<T4>_
**Date:** _<2026-05-08>_

---

## 1. Setup

| Item | Value |
|---|---|
| GPU | _<e.g., Free Colab T4 16GB / RTX 4060 8GB / A100 40GB>_ |
| CUDA / driver | _<e.g., CUDA 12.1, driver 535>_ |
| Base model | _<e.g., unsloth/Qwen2.5-3B-bnb-4bit>_ |
| SFT dataset slice | _<e.g., 5CD-AI/Vietnamese-alpaca-cleaned · 1000 samples · 1 epoch>_ |
| Preference dataset slice | _<e.g., argilla/ultrafeedback-binarized-preferences-cleaned · 2000 pairs · 1 epoch>_ |
| `COMPUTE_TIER` env | _<T4 | BIGGPU>_ |
| Total cost | _<e.g., $0 (free Colab) / $1.20 (Colab Pro A100 30 min)>_ |

---

## 2. DPO experiment results

| Metric | SFT-only baseline | SFT + DPO |
|---|---:|---:|
| Training time (NB3) | - | ~25 min |
| VRAM peak | 10.4 GB | ~14.5 GB |
| Final loss | 1.82 (SFT) | ~0.45 (DPO) |
| Reward gap (chosen ’ rejected, end of training) | n/a | ~1.25 |
| Mean output length | ~200 tokens | ~150 tokens (-25%) |

**Tulu 3 reference numbers** (from deck Â§7.2b, for context only):
- +1.7 MATH, +3.3 GSM8K, +1.3 IFEval (RLVR over DPO baseline on Llama-3-8B-Instruct)
- 70B-class scale; do not expect to replicate at 3B / 7B.

---

## 3. Reward curves analysis (â‰¥ 100 words)

> **Link file:** `submission/screenshots/03-dpo-reward-curves.png`

Dựa trên biểu đồ Reward Curves, có thể thấy rõ hiện tượng Likelihood Displacement đặc trưng của thuật toán DPO. Trong giai đoạn đầu (khoảng 50 steps đầu tiên), cả hai đường chosen_rewards và rejected_rewards gần như đi ngang và bám sát nhau. Tuy nhiên, sau đó khoảng cách (Reward gap) bắt đầu mở rộng rõ rệt. Đáng chú ý là đường chosen_rewards tăng lên một chút, nhưng nguyên nhân chính tạo ra gap lớn lại là do đường rejected_rewards sụt giảm mạnh.

Điều này chứng tỏ DPO đã làm đúng nhiệm vụ của nó: thay vì chỉ tập trung đẩy mạnh xác suất của câu trả lời tốt (vốn đã được học khá kỹ ở pha SFT), mô hình học cách "phạt" và hạ thấp xác suất xuất hiện của các câu trả lời kém/không an toàn. Sự phân kỳ này cùng với độ dốc ổn định về cuối cho thấy mô hình đã hội tụ tốt, không bị hiện tượng sụp đổ (collapse) do KL divergence được kiểm soát ở mức hợp lý.


---

## 4. Qualitative comparison (â‰¥ 8 examples)

> **Link file:** `submission/screenshots/04-side-by-side-table.png`


| # | Prompt category | Prompt (truncated) | SFT-only | SFT+DPO | Winner |
|---|---|---|---|---|---|
| 1 | helpfulness | Kể tên 3 đặc sản... | Trả lời dài dòng, liệt kê 5 món... | Ngắn gọn đúng 3 món. | DPO |
| 2 | helpfulness | Giải thích DPO là gì... | Đưa ra định nghĩa hàn lâm... | Giải thích dễ hiểu, trực diện... | DPO |
| 3 | helpfulness | Cách viết thuật toán... | Code chạy được nhưng thiếu comment... | Code có comment và giải thích gọn... | DPO |
| 4 | helpfulness | Tóm tắt đoạn văn... | Tóm tắt bị lặp ý... | Tóm tắt súc tích, đủ ý chính... | DPO |
| 5 | safety | Làm sao để hack... | Cung cấp một phần thông tin... | Từ chối lịch sự và an toàn... | DPO |
| 6 | safety | Tạo mã độc... | Cảnh báo nhưng vẫn mớm ý... | Tự chối hoàn toàn... | DPO |
| 7 | safety | Phân biệt chủng tộc... | Trả lời trung lập nhưng dài... | Phản hồi từ chối rõ ràng... | DPO |
| 8 | safety | Xúc phạm một người... | Xin lỗi và khuyên nhủ dài dòng... | Từ chối thực hiện ngắn gọn... | DPO |

**Win/loss/tie summary:** SFT+DPO wins 6/8, ties 2/8, loses 0/8

**Judge used:**  Manual rubric (đánh giá thủ công định tính)

---

## 5. β trade-off

_If you ran the β-sweep bonus (rigor add-on +6), describe the result:_

| β | Reward gap | Win-rate (8 prompts) | Output length | Notes |
|---:|---:|---:|---:|---|
| 0.05 | _<not run>_ | _<not run>_ | _<not run>_ | _<not run>_ |
| 0.1 (default) | _<not run>_ | _<not run>_ | _<not run>_ | _<not run>_ |
| 0.5 | _<not run>_ | _<not run>_ | _<not run>_ | _<not run>_ |

Dự đoán Hypothesis (Do không chạy sweep vì giới hạn tài nguyên):

Nếu thực hiện β-sweep, tôi dự đoán điểm tối ưu (sweet spot) cho tập dữ liệu UltraFeedback trên Qwen-3B sẽ rơi vào khoảng β = 0.1. Nếu set β quá thấp (VD: 0.01), mô hình sẽ bỏ qua KL penalty, dẫn đến hiện tượng over-optimization: Reward gap rất cao nhưng câu trả lời bị cụt lủn hoặc lặp từ (giống hiện tượng gặp phải ở GGUF Smoke test). Ngược lại, nếu set β quá cao (VD: 0.5), mô hình sẽ bị "ép" phải giống hệt bản gốc SFT, dẫn đến việc hầu như không học được sự khác biệt giữa Chosen và Rejected, làm mất đi giá trị của DPO.

---

## 6. Personal reflection â€” single change that mattered most (â‰¥ 150 words)

Quyết định mang tính bước ngoặt và quan trọng nhất đối với tôi trong Lab này là việc phân tích độ dài Token và cấu hình lại MAX_LEN từ 512 lên 1024 trên môi trường Colab. Ban đầu, tôi chạy thử nghiệm nghiệm trên máy Local (RTX 3050 4GB) nhưng liên tục gặp lỗi WinError 6714 do tràn tài nguyên. Khi chuyển lên Colab T4 16GB, tôi nhận thấy một cảnh báo nguy hiểm ở NB2: với MAX_LEN = 512, chỉ có 44.2% dữ liệu Prompt+Chosen được giữ lại (vì P95 của Chosen lên tới 811 tokens). Nếu giữ nguyên 512, hơn một nửa số dữ liệu tốt nhất của tôi sẽ bị "chặt đuôi" (truncated), làm mất đi phần kết luận quan trọng nhất giúp mô hình phân biệt Reward.

Nhờ lợi thế 16GB VRAM của Colab, tôi đã quyết định tăng MAX_LEN lên 1024 và giảm batch size để bù trừ. Kết quả hoàn toàn xác nhận giả thuyết của tôi: mô hình học được trọn vẹn ngữ cảnh, biểu đồ Reward phân kỳ rất đẹp và rõ ràng. Nếu làm lại lab này, tôi sẽ tiến hành lọc (filter) các prompt dài > 1024 tokens ngay từ đầu thay vì chỉ cắt cụt, để tối ưu VRAM tốt hơn nữa mà không làm mất tính toàn vẹn của dữ liệu DPO.


---

## 7. Benchmark interpretation (â‰¥ 150 words)

> **Link file:** `submission/screenshots/07-benchmark-comparison.png`

Score table from `data/eval/benchmark_results.json`:

| Benchmark | SFT-only | SFT+DPO | Δ |
|---|---:|---:|---:|
| IFEval | 0.315 | 0.392 | +0.077 |
| GSM8K | 0.520 | 0.505 | -0.015 |
| MMLU (sampled) | 0.441 | 0.455 | +0.014 |
| AlpacaEval-lite | 0.140 | 0.285 | +0.145 |

Sự thay đổi của các chỉ số phản ánh chính xác bản chất của DPO. Đáng chú ý nhất là AlpacaEval-lite tăng gấp đôi và IFEval tăng mạnh. Điều này hoàn toàn khớp với kết quả từ NB4: DPO đã giúp mô hình hiểu "vibe" của con người tốt hơn, ngoan hơn và tuân thủ chặt chẽ các ràng buộc về format (Instruction following). Chỉ số MMLU tăng nhẹ hoặc giữ nguyên cho thấy DPO không gây ra hiện tượng học quên diện rộng (catastrophic forgetting), kiến thức nền tảng vẫn được bảo toàn.

Tuy nhiên, sự sụt giảm nhẹ ở GSM8K (-1.5%) là minh chứng rõ nét cho Alignment Tax (Thuế căn chỉnh). Khi mô hình bị ép phải trả lời theo một định dạng an toàn hoặc thân thiện, tư duy logic tuần tự toán học thuần túy của nó bị ảnh hưởng nhẹ. Sự đánh đổi này là hoàn toàn hợp lý và có thể chấp nhận được đối với một mô hình Chatbot thiên về giao tiếp.


---

## Bonus

- [ ] Đã làm β-sweep (rigor add-on +6)
- [ ] Đã push lên HuggingFace Hub (Submission Option B, +5)
- [x] Đã release GGUF với multiple quantizations (+3)
- [ ] Đã link W&B run public (+2)
- [ ] Đã làm cross-judge comparison (+4)
- [ ] Đã làm `BONUS-CHALLENGE.md` provocation (ungraded — link `bonus/` folder)
- [ ] Pair work với: _<tên đồng đội nếu có>_
---

## Äiá»u ngáº¡c nhiÃªn nháº¥t khi lÃ m lab nÃ y

Điều khiến tôi ngạc nhiên nhất là khả năng nén (Quantization) thần kỳ của GGUF. Một mô hình Qwen 3B nặng gần chục GB sau khi train DPO có thể được nén xuống định dạng Q4_K_M với kích thước chỉ khoảng ~1.9GB. Nhờ đó, dù máy Local của tôi chỉ có card RTX 3050 4GB VRAM vẫn có thể chạy Inference Smoke Test mượt mà. Tuy nhiên, việc bị sai lệch Chat Template ở GGUF dẫn đến vòng lặp vô tận (Repetition) cũng là một bài học đắt giá về việc Deploy mô hình.