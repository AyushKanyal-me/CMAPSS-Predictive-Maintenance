# Phase 4 Colab GPU Handoff

Use this when local hardware is too slow for full FD001-FD004 LSTM benchmarking.

## Colab Steps

1. Upload or mount this repository in Colab.
2. Enable GPU runtime (`Runtime` -> `Change runtime type` -> `T4` or better).
3. Install dependencies.
4. Run the phase 4 command below.
5. Paste the produced metrics into `reports/colab_gpu_results.md`.

## Commands

```bash
pip install -r requirements.txt
python -m src.pipeline.multi_dataset_benchmark \
  --safe-mode \
  --include-lstm \
  --lstm-device cuda \
  --datasets FD001,FD002,FD003,FD004 \
  --skip-missing
```

## Expected Output Files

- `reports/phase4_multi_dataset_summary.csv`
- `reports/phase4_multi_dataset_summary.json`
- `reports/phase4_run_metadata.json`

## What To Share Back

- Copy metrics and run metadata into `reports/colab_gpu_results.md`.
- If any dataset failed, include traceback + command args.
