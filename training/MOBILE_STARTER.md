# AdaptiveSLM mobile training starter

Validated locally and on a free Colab T4 on 2026-10-01. This is a short pretrained-model LoRA pipeline check, not a trained phone agent. The cloud run completed ten steps and exported an adapter, merged weights and a full checkpoint; its downloaded archive passed CRC and file-hash checks. A separate earlier merged/quantized SmolLM2 smoke artifact ran on the authorized Galaxy M35 native CPU runtime. Kaggle CUDA execution and full-app/agent evaluation remain unverified. See `docs/research/colab-t4-validation-2026-10-01.json` in the repository for the cloud record.

## Use on Colab or Kaggle

1. Open `mobile_baseline.ipynb` from this archive. Upload the original `adaptive-slm-mobile-starter.zip` too; on Kaggle, attach it as an input dataset.
2. Select a GPU and enable internet. For the tested Colab path, select runtime image **2026.07** (Python 3.12 / PyTorch 2.11 / T4). Set `STARTER_ZIP` if the notebook cannot find exactly one archive.
3. Run the extraction/install cell. If dependencies were already imported, restart the kernel and rerun that cell before continuing.
4. Run CUDA validation, offline regression tests and pinned model/data preflight. Inspect filtered counts and the manifest before the ten-step training run.
5. Inspect merged-model output, then optionally convert/quantize in the separate converter environment.
6. Download `checkpoint-transfer.zip` and the desired model outputs before the ephemeral runtime ends. A zip saved only on runtime disk will be lost with that disk.

Python 3.12 was tested. The notebook retains the provider's CUDA PyTorch wheel; GPU/kernel compatibility must pass the smoke cells. This is a single-GPU recipe. Allow several GB of disk for dataset cache, HF weights, merged weights, staging and GGUF exports. It does not use hosted inference or paid teacher calls.

The Colab image's preinstalled TorchAO 0.10 breaks PEFT 0.21 even for ordinary LoRA. The tested recipe updates it to 0.17 with PyTorch 2.11; `requirements-colab-t4.txt` records that combination. Do not assume it applies to another runtime image. The T4 uses FP16: recent PyTorch probes can report emulated BF16 support, so the script explicitly requires native BF16. Precision is recorded and must match when resuming. Optional TorchAO kernels for newer GPU architectures can emit load warnings; this recipe uses ordinary LoRA and SDPA.

`colab_managed_smoke.ipynb` in the repository is a source copy of the browser-managed, GitHub-cloning run. It pins commit `a72de2b` and verifies training-file hashes. The standalone zip notebook remains the portable Colab/Kaggle entry point. Rebuild the archive after edits with `python training/build_mobile_starter.py`.

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
