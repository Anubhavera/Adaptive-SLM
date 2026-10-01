# AdaptiveSLM mobile training starter

Validated locally on 2026-10-01. This is a short pretrained-model LoRA pipeline check, not a trained phone agent. A merged/quantized SmolLM2 smoke artifact ran on the authorized Galaxy M35 native CPU runtime. Colab/Kaggle CUDA execution and full-app/agent evaluation remain unverified.

## Use on Colab or Kaggle

1. Open `mobile_baseline.ipynb` from this archive. Upload the original `adaptive-slm-mobile-starter.zip` too; on Kaggle, attach it as an input dataset.
2. Select a GPU and enable internet. Set `STARTER_ZIP` if the notebook cannot find exactly one archive.
3. Run the extraction/install cell. If dependencies were already imported, restart the kernel and rerun that cell before continuing.
4. Run CUDA validation, offline regression tests and pinned model/data preflight. Inspect filtered counts and the manifest before the ten-step training run.
5. Inspect merged-model output, then optionally convert/quantize in the separate converter environment.
6. Download `checkpoint-transfer.zip` and the desired model outputs before the ephemeral runtime ends. A zip saved only on runtime disk will be lost with that disk.

Python 3.12 was tested. The notebook retains the provider's CUDA PyTorch wheel; GPU/kernel compatibility must pass the smoke cells. This is a single-GPU recipe. Allow several GB of disk for dataset cache, HF weights, merged weights, staging and GGUF exports. It does not use hosted inference or paid teacher calls.

## Resume a planned experiment

For a longer run, choose the total step count and a new output directory before starting. Keep that configuration unchanged across sessions. Package the latest completed checkpoint with `bundle_checkpoint.py`; it includes adapter weights, optimizer, scheduler, RNG, Trainer state, the run manifest and checksums.

In the next notebook session set `CHECKPOINT_ZIP` to the uploaded archive. The restore cell checks file paths and hashes. The training script rejects incompatible configurations and resumes automatically. Hardware changes can alter numerical results; this is not a guarantee of bitwise reproducibility across GPUs. The code does not automate account switching.

## Files and scope

- `finetune_mobile.py`: pinned revisions, assistant-response labels, grouped split and resumable LoRA training/export.
- `mobile_baseline.ipynb`: setup, preflight, training, checkpoint transfer and optional conversion.
- `bundle_checkpoint.py`: portable single-device checkpoint archive.
- `prepare_legacy_converter.py`, `requirements-converter.txt`: separate tokenizer staging/environment for the old SmolLM2-compatible llama.cpp b5260 converter.
- `train.py`, `tests/test_training_regressions.py`: numerical regressions for legacy losses plus tiny offline LoRA resume/merge checks. `train.py` is included for tests; it is not the starter training entry point.

The default generic dataset was used during SmolLM2's original post-training. Validation loss here checks the pipeline, not fresh generalization. Custom tool training and evaluation require independent tasks. Native FunctionGemma/tool-result formats and newer LFM2.5/Qwen3.5 exports need separate implementations; do not substitute them into this pinned converter recipe.

Keep full failing-cell output, GPU/CUDA/PyTorch versions, `run_manifest.json` and the last persisted checkpoint if a cloud run fails. Errors are allowed to surface; there is no silent model/data fallback.
