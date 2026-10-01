#!/usr/bin/env python3
"""Stage a Transformers 5 export for the pinned llama.cpp b5260 converter.

Only tokenizer metadata changes. Model weights are linked when possible and
copied otherwise; the original HF artifact remains intact.
"""
import argparse
import json
import os
from pathlib import Path
import shutil


def stage(source, destination):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError("Use separate, non-nested source and staging directories")

    def copy_file(src, dst):
        if Path(src).suffix == ".safetensors":
            if Path(dst).exists():
                Path(dst).unlink()
            try:
                os.link(src, dst)
                return dst
            except OSError:
                pass
        return shutil.copy2(src, dst)

    shutil.copytree(source, destination, dirs_exist_ok=True, copy_function=copy_file)
    config_path = destination / "tokenizer_config.json"
    config = json.loads(config_path.read_text())
    extra = config.get("extra_special_tokens")
    if isinstance(extra, list):
        config["additional_special_tokens"] = list(dict.fromkeys(
            config.get("additional_special_tokens", []) + extra))
        del config["extra_special_tokens"]
    template = destination / "chat_template.jinja"
    if template.exists():
        config["chat_template"] = template.read_text()
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    print(f"Legacy converter input: {destination}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("destination")
    args = parser.parse_args()
    stage(args.source, args.destination)
