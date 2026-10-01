# Training guide

Start with the pretrained-model LoRA baseline in [mobile_baseline.ipynb](../training/mobile_baseline.ipynb). The custom student remains experimental. Its cloud memory use and model quality are not validated; the old TPU, training-duration and 512 MB claims should not guide a run. Read the [audit](research/2026-10-01-audit-and-plan.md) for the reasons.

## First Colab or Kaggle run

1. Upload `artifacts/adaptive-slm-mobile-starter.zip` and open `training/mobile_baseline.ipynb`. On Kaggle, add the zip as an input dataset.
2. Select a GPU runtime and enable internet. The notebook checks that the provider's CUDA PyTorch wheel can execute on the assigned GPU. This starter uses one GPU; it does not use a second GPU or TPU.
3. Install pinned dependencies. Restart the kernel if those packages were already imported. The notebook retains the provider's PyTorch wheel and records its version.
4. Run source/data preflight, inspect the manifest and filtered sample counts, then run the ten-step training smoke test.
5. Inspect the merged HF model. Optionally convert it with the isolated, pinned b5260 converter and quantize to Q4_K_M.
6. Save outputs to persistent storage before ending the session. Local regression tests and CPU export checks passed; the actual free cloud GPU must still pass this notebook.

The default SmolLM2-360M model and smol-smoltalk dataset are pinned to immutable revisions. Preflight downloads a local dataset snapshot before sampling; allow several GB of cache and export space. A generic-data smoke test establishes pipeline operation, not agent competence. The source dataset was also used in the model's original post-training and cannot establish fresh held-out generalization.

## Continue across free sessions

Decide the total `--max-steps` before starting a research run. The ten-step notebook is a separate smoke experiment. For a longer experiment, use a new output directory and a larger planned step count. Save every five or twenty-five steps according to the amount of work you can afford to repeat.

The latest complete checkpoint includes adapter weights, optimizer, scheduler, RNG and Trainer state. **An adapter or merged model alone is insufficient to resume the same run.** Package it after a checkpoint save completes:

```bash
python training/bundle_checkpoint.py /path/to/run /path/to/checkpoint-transfer.zip
```

Download the archive or copy it to persistent storage. In another session, upload the starter and archive, set `CHECKPOINT_ZIP` in the notebook, and use the same model/data revisions, tokenization, seed, batch size, accumulation, learning rate, total steps and pinned package environment. The notebook verifies archive hashes before extracting. The script rejects incompatible manifests and resumes automatically. An already-complete run evaluates and exports without another training step.

Interruptions lose work since the last persisted checkpoint. Ephemeral checkpoints disappear with the runtime even if they were saved successfully. Different GPU models can change numerical results; restoring state does not guarantee bitwise identity across hardware. Keep a fixed GPU type for final reproducibility measurements where possible. These scripts store no service login or credentials.

## Custom app-tool data

Use JSONL records with `messages`: system/user context and a final assistant target. This starter accepts string-valued system, user and assistant messages; represent a tool request as assistant JSON. It does not yet support native `tool` role conversations. Only the final assistant response receives labels. Overlength records are counted and skipped intact. Identical user-question groups stay in one split across personas and alternative answers.

Define the tool schema, negative examples and independent held-out tasks before adapting. Later work must support native tool/result templates, multi-step state and constrained decoding. Increasing training steps cannot substitute for this design.

## If a run fails

Keep the full failing cell output, `run_manifest.json`, GPU name, CUDA/PyTorch versions and last completed checkpoint. Do not replace data, alter dependencies or change teacher precision silently. Conversion uses a separate environment because b5260 dependencies conflict with the new training stack. LFM2.5 and Qwen3.5 need a modern runtime/export path; the starter's b5260 conversion is specifically for SmolLM2.
