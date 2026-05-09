# DeepLearning_Final
# Parameter-Efficient Multimodal Distillation for Science Multiple-Choice Reasoning

This repository contains the training and inference code for our submission to the multimodal scientific multiple-choice reasoning task. We adapt **SmolVLM-500M-Instruct** as a student model under a **5M trainable parameter** budget, supervising it with confidence-gated soft labels distilled from a **Qwen2.5-VL-7B** teacher. Our final submission, an ensemble of two LoRA students trained with and without teacher-generated image descriptions, achieves **84.4% accuracy** on the public leaderboard.

## Authors

- **Yashwanth Kasanneni** ([yk3456@nyu.edu](mailto:yk3456@nyu.edu))
- **Laya Mangalagiri** ([lm5809@nyu.edu](mailto:lm5809@nyu.edu))

New York University

## Overview of approach

The pipeline has three stages:

1. **Teacher inference.** The Qwen2.5-VL-7B teacher generates per-choice probability distributions for the entire training set. We use the worked-solution column from the dataset only when generating these training-time soft labels (it is unavailable at test time and is never seen by the student). On validation we generate a second set of soft labels without the solution leak as an honest evaluation of the teacher's standalone capability.
2. **Student fine-tuning.** Two SmolVLM students are fine-tuned with LoRA (rank 8, attention + MLP projections, ≈4.78M trainable parameters). The training loss combines hard cross-entropy with a temperature-softened KL divergence to the cleaned teacher distribution. The two students differ in input format: one sees only the original prompt; the other additionally receives a teacher-generated text description of the image.
3. **Ensemble inference.** At test time, both students' softmaxed choice probabilities are averaged with equal weights, then argmaxed over the valid choice set.

## Repository contents

### Notebooks

| File | Role |
|---|---|
| `DL_Final_v2.ipynb` | End-to-end pipeline for the no-caption student. Generates teacher soft labels, builds the student prompt, fine-tunes the LoRA adapter, and runs evaluation. |
| `DL_final_v5.ipynb` | Same pipeline plus image-description generation, the caption-augmented student, and ensemble inference. Loads both students as separate adapters on a shared base model. |

### Trained adapters

| Directory | Description | Used in submission? |
|---|---|---|
| `smolvlm_kd_lora_best_v4_fixed/` | The no-caption student. LoRA adapter weights only (the base SmolVLM is downloaded at runtime from Hugging Face). | ✅ Yes |
| `smolvlm_kd_lora_best_v5/` | The caption-augmented student. | ✅ Yes |
| `smolvlm_kd_lora_best_v6_dora/` | DoRA variant; included for completeness. Did not improve the ensemble in our experiments. | ❌ No |

### Cached intermediate artifacts

| File | Description |
|---|---|
| `image_captions_v5.csv` | Teacher-generated text description for every image in train, val, and test (5,165 rows). Used by the caption-augmented student. |
| `teacher_train_probs_logits_v2_new.csv` | Teacher per-choice probabilities, raw logits, and confidence values on the training set. Generated with the worked-solution column included in the teacher prompt to maximize label quality. |
| `teacher_val_probs_logits_v2.csv` | Teacher per-choice probabilities, raw logits, and confidence values on the validation set. Generated **without** the worked solution; this is the honest teacher evaluation (81.58% accuracy). |

These cached artifacts let you skip the most expensive parts of the pipeline (teacher inference and caption generation) when reproducing student training.

## Setup

The notebooks were developed in Google Colab Free tier (single NVIDIA T4 GPU, 16 GB VRAM). They expect a Google Drive directory at `/content/drive/MyDrive/DL_Final_DATA/` containing:

- The provided dataset CSVs: `train (1).csv`, `val.csv`, `test (1).csv`
- The image folder (referenced by the `image_path` column in each CSV)
- Optionally, the cached artifacts and adapter directories from this repository, to skip teacher inference and caption generation

### Dependencies

Installed at the top of each notebook:

```bash
pip install -q transformers accelerate bitsandbytes peft datasets pillow pandas tqdm
```

Tested with PyTorch 2.x, Transformers 4.x, and PEFT 0.x (Colab defaults at the time of writing).

### Hugging Face authentication

Loading the Qwen2.5-VL teacher requires Hugging Face authentication. Set the `HF_TOKEN` Colab secret to a token with read access.

## How to reproduce

### Path A: Run the full pipeline from scratch

1. Open `DL_Final_v2.ipynb` in Colab.
2. Mount Drive and confirm the dataset CSVs are present at `DATA_DIR`.
3. Run cells top to bottom:
   - Teacher loads in 4-bit and runs inference on the training set (with worked solution) and validation set (without). Soft labels are saved to `teacher_train_probs_logits_v2_new.csv` and `teacher_val_probs_logits_v2.csv` with checkpointing every 500 rows. Allow 1–2 hours per split.
   - Confidence-gated cleanup replaces teacher labels with one-hot ground truth wherever the teacher is wrong or below 0.7 confidence (≈6.4% of training rows in our run).
   - Teacher is freed from GPU memory.
   - SmolVLM student loads with LoRA (rank 8, attention + MLP). The notebook prints trainable parameter count to verify it is below the 5M cap.
   - Training runs for 8 epochs with cosine LR (peak 2e-4, 5% warmup). Best adapter by validation accuracy is saved to `smolvlm_kd_lora_best_v4_fixed/`.
4. Open `DL_final_v5.ipynb` to add image descriptions and the caption-augmented student.
   - Caption generation: ~3.3 hours total for all 5,165 images. Resumable from `image_captions_v5.csv` if it already exists.
   - Caption-augmented student trains for 7 epochs with the same recipe; saves to `smolvlm_kd_lora_best_v5/`.
   - Final cells load both adapters on a shared base model and produce the ensemble submission.

### Path B: Skip to student training using cached artifacts

If you have already downloaded the cached files from this repository, you can skip both the teacher inference (~3 hours) and the caption generation (~3.3 hours):

1. Place `teacher_train_probs_logits_v2_new.csv`, `teacher_val_probs_logits_v2.csv`, and `image_captions_v5.csv` in `DATA_DIR`.
2. Open `DL_Final_v2.ipynb` and skip the teacher inference cells; jump directly to the soft-target cleanup and student training cells.
3. For the caption-augmented student, open `DL_final_v5.ipynb` and skip the caption-generation cells; jump directly to the merging and student training cells.

### Path C: Inference only with the trained adapters

If you only want to run the final submission ensemble using the pre-trained adapters in this repository:

1. Download `smolvlm_kd_lora_best_v4_fixed/`, `smolvlm_kd_lora_best_v5/`, and `image_captions_v5.csv` to your `DATA_DIR`.
2. Open the inference section near the bottom of `DL_final_v5.ipynb`.
3. The cells load the base SmolVLM, attach both LoRA adapters via PEFT (`PeftModel.from_pretrained` for the first, `model.load_adapter` for the second), build both prompt formats per test row, run each adapter, average their softmaxed choice logits with equal weights, and write the submission CSV.

## Implementation notes

A few details that matter for getting the pipeline to work correctly:

- **Left padding** on the student tokenizer is required so that the last position of every batched sequence is the actual end of the prompt (after `Answer:`), not a pad token. With right padding, logits at pad positions are read for shorter examples in mixed-length batches, which silently degrades accuracy.
- **Digit-token variant aggregation.** For each digit 0–4 we collect every single-token vocabulary entry that decodes to that digit (`'0'`, `' 0'`, `'\n0'`, etc.) and aggregate via log-sum-exp before applying softmax across choices. This is done identically for both the teacher and the student, so the supervision signal and the inference signal are compatible.
- **Choice masking.** All loss and prediction operations mask out the choice slots that don't exist for a given example (e.g. positions 3 and 4 of a 3-choice question), by setting their logits to a large negative number before softmax/argmax.
- **Confidence-gated cleanup.** Where the teacher is wrong or below 0.7 confidence, the soft target is replaced with a one-hot of the ground-truth label. This prevents the student from being trained to confidently mimic teacher mistakes.
- **Loss weighting.** The cross-entropy term is weighted at 0.7 and the KL term at 0.3. We weight CE more than KL because the teacher's standalone validation accuracy (81.58%) is below the student's eventual accuracy, so a model that defers heavily to the teacher cannot reliably exceed it.

## Results

| System | Validation | Public leaderboard |
|---|---:|---:|
| Student A (no captions) | 81.11% | 83.8% |
| Student B (caption-augmented) | 80.44% | 83.9% |
| Student C (DoRA variant) | 78.24% | — |
| **Ensemble (Student A + Student B, 50/50)** | **82.44%** | **84.4%** |

Final submission: 84.4% on the public leaderboard.

## Acknowledgments

We thank the course staff for the assignment and the leaderboard infrastructure.
