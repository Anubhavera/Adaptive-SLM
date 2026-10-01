#!/usr/bin/env python3
"""Rebuild the portable starter from the current source, with checksums."""
import hashlib
import json
from pathlib import Path
import zipfile


def main():
    root = Path(__file__).resolve().parent
    output = root.parent / "artifacts" / "adaptive-slm-mobile-starter.zip"
    names = [
        "finetune_mobile.py", "requirements-mobile.txt", "requirements-colab-t4.txt",
        "requirements-converter.txt", "prepare_legacy_converter.py",
        "bundle_checkpoint.py", "mobile_baseline.ipynb", "train.py",
        "tests/test_training_regressions.py",
    ]
    contents = {name: (root / name).read_bytes() for name in names}
    contents["README.md"] = (root / "MOBILE_STARTER.md").read_bytes()
    checksums = {name: hashlib.sha256(data).hexdigest() for name, data in contents.items()}
    contents["starter_sha256.json"] = (json.dumps(checksums, indent=2) + "\n").encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(contents.items()):
            info = zipfile.ZipInfo(name, date_time=(2026, 10, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, data)
    with zipfile.ZipFile(output) as archive:
        assert archive.testzip() is None
        for name, sha in checksums.items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == sha
    sha = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_suffix(".zip.sha256").write_text(f"{sha}  {output.name}\n")
    print(f"Built {output}: {output.stat().st_size} bytes; SHA256 {sha}")


if __name__ == "__main__":
    main()
