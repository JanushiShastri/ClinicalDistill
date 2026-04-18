# ClinicalDistill Research Log

## Experiment 1 — Gemma-3-1B LoRA
- Date: April 13, 2026
- Model: google/gemma-3-1b-it
- Method: LoRA (r=8, alpha=16)
- Dataset: 120 train / 30 test
- Epochs: 5
- Batch size: 2
- Learning rate: 2e-4
- Final loss: 0.73
- Training time: ~2 min 17 sec
- Hardware: Google Colab T4 GPU
- Notes: Loss dropped from 2.57 → 0.73. Clean run, no crashes.

## Observation 1 — Output Format Inconsistency
- Model produces inconsistent JSON structure across examples
- Root cause: training data has mixed output formats
- Fix for next run: normalize all train.jsonl outputs to exact schema
  before retraining

  ## Dataset Fix — April 13 2026
- All 120 examples confirmed flat format ✅
- Fixed duration/severity array length mismatch
- urgent balance: 44 true / 76 false (37/63 split)
- Files: train_fixed.jsonl, test_fixed.jsonl
- Use these for all future training runs

## Evaluation — Gemma-3-1B LoRA (Experiment 1)
- Valid JSON rate : 60.0% (18/30)
- Avg Symptom F1  : 0.648
- Urgent Accuracy : 33.3% (6/18)
- Dataset used    : train.jsonl (original)
- Notes           : Array misalignment identified and fixed
                    Next run will use train_fixed.jsonl


                    ## Experiment 1 — Gemma-3-1B LoRA (Original Dataset)
- Date: April 13, 2026
- Model: google/gemma-3-1b-it
- Method: LoRA (r=8, alpha=16, target: q_proj + v_proj)
- Dataset: train.jsonl (120 train / 30 test, GPT-4o generated)
- Batch size: 2 (gradient accumulation: 4)
- Learning rate: 2e-4
- Hardware: Google Colab T4 GPU

| Epochs | Valid JSON | Symptom F1 | Urgent Accuracy |
|--------|------------|------------|-----------------|
| 5      | 60.0%      | 0.648      | 33.3%           |
| 10     | 66.7%      | 0.658      | 30.0%           |

### Issues Found
- Duration/severity arrays misaligned with symptoms length
- More epochs not helping — bottleneck is prompt clarity
- Urgent accuracy weak — needs stricter prompt

### Fix Applied
- Fixed array alignment → train_fixed.jsonl / test_fixed.jsonl
- All 120 examples confirmed flat format
- Urgent balance: 44 true / 76 false (healthy 37/63 split)

### Next Run
- Use train_fixed.jsonl
- Stricter prompt with explicit schema
- Target: Valid JSON >80%, F1 >0.70

---

## Experiment 2 — Gemma-3-1B LoRA (Fixed Dataset + Casual Examples)
- Date: April 16, 2026
- Notebook: ClinicalDistill_LoRA_Gemma.ipynb
- Model: google/gemma-3-1b-it (999,885,952 parameters)
- Method: LoRA (r=16, alpha=32, target: q_proj + v_proj, dropout=0.05)
- Trainable params: 1,490,944 (0.1489% of total)
- Dataset: train_fixed.jsonl (145 train / 35 test)
  - 120 formal clinical (GPT-4o) + 25 casual/colloquial (Claude)
  - Domains: cardiac, respiratory, neurological, gastrointestinal
- Epochs: 7
- Batch size: 2 (gradient accumulation: 4, effective batch: 8)
- Learning rate: 2e-4
- Hardware: Google Colab T4 GPU
- Prompt format: XML tags (`<instruction>`, `<input>`, `<output>`)

### Evaluation Results

| Metric           | Score         |
|------------------|---------------|
| Valid JSON rate  | 100% (35/35)  |
| Avg Symptom F1   | 0.781         |
| Urgent Accuracy  | 85.7% (30/35) |

### Improvement over Experiment 1

| Metric          | Exp 1 (broken data) | Exp 2 (fixed data) | Delta   |
|-----------------|---------------------|--------------------|---------|
| Valid JSON      | 60.0%               | 100.0%             | +40.0%  |
| Symptom F1      | 0.648               | 0.781              | +0.133  |
| Urgent Accuracy | 33.3%               | 85.7%              | +52.4%  |

### Key drivers of improvement
- Fixed duration/severity array alignment (parallel to symptoms)
- Added 25 casual training examples — handles patient-reported language
- XML prompt format gives cleaner output boundary
- r=16 vs r=8 — doubled LoRA rank

### Remaining issues
- Symptom duplication: modifiers treated as separate symptoms ("hurts when sitting" → extra entry)
- 5 urgent misclassifications out of 35
- 7 epochs likely slight overfit — try 5 epochs next run

### Next: Experiment 3 — Gemma-3-1B QLoRA
- Same dataset (train_fixed.jsonl 145/35), same LoRA config
- Add 4-bit quantization via BitsAndBytesConfig (nf4)
- Track: training time, GPU memory, F1 vs LoRA baseline

---

## Experiment 3 — Gemma-3-1B QLoRA
- Date: April 17, 2026
- Notebook: ClinicalDistill_QLoRA_Gemma.ipynb
- Model: google/gemma-3-1b-it (999,885,952 parameters)
- Method: QLoRA — 4-bit quantization (nf4, double quant) + LoRA (r=16, alpha=32, q_proj + v_proj)
- Dataset: train_fixed.jsonl (145 train / 35 test) — identical to Exp 2
- Epochs: 5
- Batch size: 2 (gradient accumulation: 4, effective batch: 8)
- Learning rate: 2e-4
- Precision: bf16 (fp16 conflicts with Gemma-3 BFloat16 internals under 4-bit quant)
- Hardware: Google Colab T4 GPU

### Evaluation Results

| Metric           | Score         |
|------------------|---------------|
| Valid JSON rate  | 100% (35/35)  |
| Avg Symptom F1   | 0.740         |
| Urgent Accuracy  | 82.9% (29/35) |

### LoRA vs QLoRA Comparison — Gemma-3-1B

| Metric          | LoRA (Exp 2) | QLoRA (Exp 3) | Delta   |
|-----------------|--------------|---------------|---------|
| Valid JSON      | 100%         | 100%          | 0       |
| Symptom F1      | 0.781        | 0.740         | -0.041  |
| Urgent Accuracy | 85.7%        | 82.9%         | -2.8%   |
| Epochs          | 7            | 5             | —       |

### Key findings
- F1 drop of 0.041 — quantization has minimal accuracy cost (< 5%)
- Valid JSON remains 100% — output format unaffected by quantization
- Urgent accuracy drops slightly (1 more misclassification)
- bf16 required instead of fp16 — Gemma-3 uses BFloat16 internally; fp16 gradient scaling throws NotImplementedError under 4-bit quant

### Why LoRA outperforms QLoRA — and why that's expected
- LoRA trains adapters on full float16 weights — every weight value is precise
- QLoRA trains adapters on 4-bit compressed weights — some information is lost in compression
- LoRA adapters in QLoRA must compensate for slightly degraded base representations
- A ~5% F1 drop is consistent with published QLoRA results on other tasks

### Paper argument — deployment tradeoff
- QLoRA retains 94.8% of LoRA accuracy (0.740 vs 0.781) at ~25% of the memory cost
- For resource-limited settings (small clinic GPU, edge deployment), a model that runs beats a model that doesn't
- QLoRA is the preferred option when VRAM is constrained; LoRA is preferred when accuracy is the priority
- This tradeoff analysis is the practical contribution — helps practitioners make deployment decisions

### Next: Experiment 4 — Phi-2 LoRA
- Model: microsoft/phi-2 (2.7B parameters)
- Same dataset, same LoRA config (r=16, alpha=32)
- Same eval — compare F1, urgent accuracy, training time across model families