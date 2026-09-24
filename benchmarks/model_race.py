#!/usr/bin/env python3
"""
Multi-Model Benchmark Race
Run the AdaptiveSLM bench against any set of GGUF models and pick a winner.

Usage:
    python benchmarks/model_race.py --models models/qwen*.gguf models/gemma*.gguf
    python benchmarks/model_race.py  # auto-discovers all .gguf in models/
"""

import subprocess
import re
import sys
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

BENCH_BIN_CANDIDATES = [
    "core/build/adaptive_slm_bench.exe",
    "core/build/adaptive_slm_bench",
]

@dataclass
class ModelResult:
    model_name: str
    model_path: str
    file_size_mb: float
    tokens_per_sec: float = 0.0
    avg_latency_ms: float = 0.0
    peak_ram_mb: float = 0.0
    state_mb: float = 0.0
    passed_memory_target: bool = False
    passed_speed_target: bool = False
    fake_output_detected: bool = False
    output_previews: list = field(default_factory=list)
    raw_output: str = ""
    error: Optional[str] = None

def find_bench_binary() -> Optional[Path]:
    for candidate in BENCH_BIN_CANDIDATES:
        p = Path(candidate)
        if p.exists():
            return p
    return None

def parse_bench_output(output: str) -> dict:
    data = {}

    # tok/s — "X.X tok/s" pattern
    for m in re.finditer(r'([\d.]+)\s*tok/s', output):
        val = float(m.group(1))
        data['tokens_per_sec'] = max(data.get('tokens_per_sec', 0.0), val)

    # Overall tokens per second line
    m = re.search(r'Overall tokens per second:\s*([\d.]+)', output)
    if m:
        data['tokens_per_sec'] = float(m.group(1))

    # Latency — "X.X ms"
    latencies = [float(m.group(1)) for m in re.finditer(r'in\s+([\d.]+)\s*ms', output)]
    if latencies:
        data['avg_latency_ms'] = sum(latencies) / len(latencies)

    # RAM — Model + state size
    m = re.search(r'Model \+ state size:\s*([\d.]+)\s*MB', output)
    if m:
        data['peak_ram_mb'] = float(m.group(1))

    m = re.search(r'Current state size:\s*([\d.]+)\s*MB', output)
    if m:
        data['state_mb'] = float(m.group(1))

    # Output previews
    previews = re.findall(r'Output:\s*(.+?)(?:\.\.\.)?$', output, re.MULTILINE)
    data['previews'] = previews[:3]

    # Fake output detection
    data['fake'] = 'Demo response' in output

    return data

def run_model(bench: Path, model_path: Path) -> ModelResult:
    name = model_path.stem
    size_mb = model_path.stat().st_size / (1024 * 1024)
    result = ModelResult(model_name=name, model_path=str(model_path), file_size_mb=size_mb)

    print(f"\n{'='*60}")
    print(f"  Testing: {name}")
    print(f"  File size: {size_mb:.0f} MB")
    print(f"{'='*60}")

    try:
        proc = subprocess.run(
            [str(bench), str(model_path)],
            capture_output=True, text=True, timeout=300
        )
        output = proc.stdout + proc.stderr
        result.raw_output = output
        print(output)

        parsed = parse_bench_output(output)
        result.tokens_per_sec = parsed.get('tokens_per_sec', 0.0)
        result.avg_latency_ms = parsed.get('avg_latency_ms', 0.0)
        result.peak_ram_mb = parsed.get('peak_ram_mb', 0.0)
        result.state_mb = parsed.get('state_mb', 0.0)
        result.output_previews = parsed.get('previews', [])
        result.fake_output_detected = parsed.get('fake', False)
        result.passed_memory_target = 0 < result.peak_ram_mb < 512
        result.passed_speed_target = result.tokens_per_sec > 40

        if result.fake_output_detected:
            result.error = "FAKE OUTPUT — inference is not real"

    except subprocess.TimeoutExpired:
        result.error = "Timed out (>300s)"
    except Exception as e:
        result.error = str(e)

    return result

def print_comparison(results: list[ModelResult]):
    print("\n" + "=" * 90)
    print("  BENCHMARK RACE RESULTS")
    print("=" * 90)

    fmt = "{:<35} {:>10} {:>10} {:>12} {:>10} {:>8}"
    print(fmt.format("Model", "Tok/s", "Latency ms", "RAM (MB)", "File (MB)", "Status"))
    print("-" * 90)

    ranked = sorted(results, key=lambda r: r.tokens_per_sec, reverse=True)
    for i, r in enumerate(ranked):
        status = "ERROR" if r.error else ("FAKE" if r.fake_output_detected else "OK")
        if i == 0 and not r.error and not r.fake_output_detected:
            status = "WINNER"
        ram = f"{r.peak_ram_mb:.0f}" if r.peak_ram_mb > 0 else "N/A"
        lat = f"{r.avg_latency_ms:.0f}" if r.avg_latency_ms > 0 else "N/A"
        print(fmt.format(
            r.model_name[:35],
            f"{r.tokens_per_sec:.1f}",
            lat,
            ram,
            f"{r.file_size_mb:.0f}",
            status
        ))

    print("=" * 90)

    winner = next((r for r in ranked if not r.error and not r.fake_output_detected), None)
    if winner:
        print(f"\n  WINNER: {winner.model_name}")
        print(f"    Speed:   {winner.tokens_per_sec:.1f} tok/s")
        print(f"    RAM:     {winner.peak_ram_mb:.0f} MB")
        print(f"    File:    {winner.file_size_mb:.0f} MB")
        if winner.output_previews:
            print(f"    Sample:  {winner.output_previews[0][:100]}...")
        print(f"\n  Ship this model: {winner.model_path}")

    print()

def main():
    parser = argparse.ArgumentParser(description="Multi-model benchmark race")
    parser.add_argument("--models", nargs="*", help="GGUF model paths (default: auto-discover models/)")
    parser.add_argument("--bench", help="Path to bench binary (default: auto-detect)")
    args = parser.parse_args()

    bench = Path(args.bench) if args.bench else find_bench_binary()
    if not bench:
        print("ERROR: bench binary not found. Build first:")
        print("  cd core/build && cmake --build .")
        sys.exit(1)

    if args.models:
        model_paths = [Path(m) for m in args.models]
    else:
        models_dir = Path("models")
        model_paths = sorted(models_dir.glob("*.gguf"))

    model_paths = [p for p in model_paths if p.exists()]

    if not model_paths:
        print("No GGUF models found. Download one first:")
        print("  hf download Qwen/Qwen2.5-0.5B-Instruct-GGUF qwen2.5-0.5b-instruct-q4_k_m.gguf --local-dir models/")
        sys.exit(1)

    print(f"Racing {len(model_paths)} model(s)...")

    results = [run_model(bench, p) for p in model_paths]
    print_comparison(results)

if __name__ == "__main__":
    main()
