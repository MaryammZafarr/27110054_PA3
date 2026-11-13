# ATML PA3 (27110054): fine-tuning, vision-language grounding and test-time compute

Code, configurations, split / example IDs and machine-readable results for the three tasks. Seed 6304 is used throughout. All runs were done on one Tesla T4 (Kaggle).

| Task | Code | Results |
|---|---|---|
| 1. Qwen3-0.6B fine-tuning on MetaMathQA (Full FT, L2-SP, LoRA-4/16, OPLoRA-16) | `task1/` | `results/task1/` |
| 2. SigLIP + Qwen3-0.6B vision-language model and grounding evaluation | `task2/` | `results/task2/` |
| 3. Test-time compute (sampling, selection, adaptive stopping, live refinement) | `task3/` | `results/task3/` |

`kaggle/` holds the Kaggle runner scripts that executed the commands below (paths there are Kaggle paths). Raw datasets, model weights, LoRA adapters, connector weights, cached features, images and the Task 3 pickles are **not** committed.

## Environment
`pip install -r requirements.txt` (Python 3.10+, one CUDA GPU). `math-verify==0.9.0`, `lm_eval==0.4.13` and `torch==2.11.0` are the versions recorded in the result files; the other packages are lower bounds (the Kaggle image supplied the exact builds, which were not pinned).

## Reproducing each task
Every script takes `--out <dir>`; later steps read what earlier steps wrote there.

**Task 1** (data: MetaMathQA from the Hugging Face hub; model: Qwen3-0.6B)
```
python task1/selftest.py
python task1/prepare_data.py --out OUT --qual_yaml task1/first_plot_questions.yaml
for m in full_ft l2sp lora4 lora16 oplora16; do python task1/train.py --method $m --out OUT; done   # add --l2sp_ref_device cpu for l2sp
python task1/evaluate.py --out OUT
python task1/diagnostics.py --out OUT
python task1/analyze.py --out OUT
python task1/verify.py --out OUT
```
`train.py` writes rolling recovery checkpoints and resumes with `--resume`. `task1/verify_adapters.py` is an extra check on the exported adapters.

**Task 2** (data: COCO 2014 captions / instances / images and VQAv2 annotations, downloaded by `prepare_data.py`)
```
python task2/selftest.py
python task2/prepare_data.py --out OUT --raw_dir RAW
python task2/cache_features.py --out OUT --delete_train_images
python task2/train.py --out OUT --run main --tokens 196 --resume
python task2/evaluate.py --out OUT --run main
python task2/analyze.py --out OUT --run main
python task2/verify.py --out OUT --run main
```
The optional 49-token compression run (`--tokens 49`) was not run.

**Task 3** (data: the supplied `ATML-PA3-Data` rollouts, not committed)
```
python task3/selftest.py
python task3/prepare.py --data DATA --out OUT
python task3/offline.py --out OUT
python task3/live.py --data DATA --out OUT          # needs a GPU; 50 MATH questions
python task3/analyze.py --out OUT
python task3/verify.py --out OUT --part offline     # and --part live
```

## Split and example IDs
- Task 1: `results/task1/splits/` (training IDs, probe set, qualitative prompts), `results/task1/data/` (excluded IDs, training statistics)
- Task 2: `results/task2/splits/` (caption, grounding and VQA train / dev / test splits, mismatch mapping, `selection_stats.json`), label audit in `results/task2/audit/`
- Task 3: `results/task3/offline/splits/` (200 development / 800 held-out question IDs per dataset, ten saved reveal orders, live-question IDs)

## Results
Every table and figure in the report is produced by `analyze.py` of the corresponding task and is saved as CSV / Markdown / TeX / PNG / PDF:
- `results/task1/report/`, `results/task2/report_t2/`, `results/task3/offline/report_t3/` and `results/task3/live/report_t3/`
- per-example outputs: `results/task1/eval/*/` (lm-eval samples and scores), `results/task2/eval/main/predictions.jsonl`, `results/task3/offline/offline/` (per-question selections, stopping statistics), `results/task3/live/live/calls.jsonl` (all live prompts, responses, seeds, lengths and truncation flags)
- run summaries and training logs: `results/task1/runs/*/`, `results/task2/runs/main/`
- verification outputs: `results/task1/verify_report.json` (see `VERIFY_NOTE.md` for the one failing check), `results/task2/verify_t2_main.json`, `results/task3/*/verify_t3_*.json`

## Attribution
- Evaluation: [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) (EleutherAI), used as a library for Task 1; [math-verify](https://github.com/huggingface/Math-Verify) (Hugging Face) for MATH answer equivalence in Task 3.
- `task2/common.py`: port of the official VQA evaluation answer normalisation (`vqaEval.py`, VQA team, BSD licence).
- `task3/common.py`: port of MathVista's `normalize_extracted_answer` (`evaluation/calculate_score.py`, [lupantech/MathVista](https://github.com/lupantech/MathVista)).
- Task 3 rollouts, `README.MD` field definitions and model revision were supplied with the assignment.
- Models and data: Qwen3-0.6B, Qwen2.5-3B-Instruct, Qwen2.5-VL-3B-Instruct (supplied rollouts), SigLIP base patch16-224, MetaMathQA, COCO 2014, VQAv2, MATH, MathVista.
