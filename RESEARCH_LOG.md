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