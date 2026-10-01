#!/usr/bin/env python3
"""Package the latest complete single-device Trainer checkpoint for transfer.

Run after a checkpoint save completes. Upload/download the resulting archive
through your chosen persistent storage; no account credentials are needed here.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import tempfile
import zipfile


def bundle(run, output):
    run, output = Path(run).resolve(), Path(output).resolve()
    candidates = [(int(p.name.split("-")[1]), p) for p in run.iterdir()
                  if p.is_dir() and re.fullmatch(r"checkpoint-\d+", p.name)]
    if not candidates:
        raise ValueError("No checkpoint found; wait for the first completed save")
    _, checkpoint = max(candidates)
    required = [run / "run_manifest.json"] + [checkpoint / name for name in
                ["trainer_state.json", "optimizer.pt", "scheduler.pt", "rng_state.pth", "adapter_config.json"]]
    for path in required:
        if not path.is_file():
            raise ValueError(f"Incomplete single-device checkpoint: {path}")
    if not list(checkpoint.glob("adapter_model.*")):
        raise ValueError("Checkpoint is missing adapter weights")
    if output == run / "run_manifest.json" or checkpoint in output.parents:
        raise ValueError("Place the archive outside the checkpoint directory")
    files = [run / "run_manifest.json"] + sorted(p for p in checkpoint.rglob("*") if p.is_file())
    output.parent.mkdir(parents=True, exist_ok=True)
    hashes = {}
    with tempfile.NamedTemporaryFile(dir=output.parent, suffix=".zip", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED) as archive:
            for path in files:
                name = path.relative_to(run).as_posix()
                sha = hashlib.sha256()
                with path.open("rb") as source, archive.open(name, "w", force_zip64=True) as target:
                    while chunk := source.read(1024 * 1024):
                        sha.update(chunk)
                        target.write(chunk)
                hashes[name] = sha.hexdigest()
            archive.writestr("checkpoint_sha256.json", json.dumps(hashes, indent=2) + "\n")
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Transfer archive: {output}; checkpoint: {checkpoint.name}; bytes: {output.stat().st_size}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run")
    parser.add_argument("output")
    args = parser.parse_args()
    bundle(args.run, args.output)
