#!/usr/bin/env python3
"""Read phone hardware, battery and thermal state through authorized ADB.

No installs, settings changes, screenshots, messages, or app data reads.
"""
import argparse
import datetime
import json
from pathlib import Path
import shutil
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial", help="Select one device when several are connected")
    parser.add_argument("--output", default="docs/research/phone-profile.json")
    args = parser.parse_args()
    adb = shutil.which("adb")
    if not adb:
        raise SystemExit("ADB not found. Install Android SDK platform-tools and put adb on PATH.")
    listing = subprocess.run([adb, "devices"], check=True, capture_output=True, text=True, timeout=15).stdout
    devices = [line.split()[:2] for line in listing.splitlines()[1:] if len(line.split()) >= 2]
    authorized = [serial for serial, state in devices if state == "device"]
    if args.serial:
        if args.serial not in authorized:
            raise SystemExit("Selected device is not authorized. Unlock the phone and accept the USB debugging prompt.")
        serial = args.serial
    elif len(authorized) == 1:
        serial = authorized[0]
    else:
        raise SystemExit("Need one authorized phone, or --serial to choose one. Unlock it and accept the USB debugging prompt.")

    def shell(*command):
        result = subprocess.run([adb, "-s", serial, "shell", *command], capture_output=True, text=True, timeout=20)
        return result.stdout.strip() if result.returncode == 0 else {"unavailable": result.stderr.strip()}

    properties = ["ro.product.manufacturer", "ro.product.model", "ro.product.device",
                  "ro.build.version.release", "ro.build.version.sdk", "ro.product.cpu.abilist",
                  "ro.soc.manufacturer", "ro.soc.model", "ro.board.platform", "ro.hardware"]
    report = {"captured_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "properties": {prop: shell("getprop", prop) for prop in properties}}
    mem = shell("cat", "/proc/meminfo")
    report["memory"] = {line.split(":", 1)[0]: line.split(":", 1)[1].strip()
                        for line in mem.splitlines() if line.startswith(("MemTotal:", "MemAvailable:"))} if isinstance(mem, str) else mem
    battery = shell("dumpsys", "battery")
    report["battery"] = {line.strip().split(":", 1)[0]: line.split(":", 1)[1].strip()
                         for line in battery.splitlines() if ":" in line and line.strip().split(":", 1)[0] in
                         {"level", "scale", "temperature", "status", "USB powered", "AC powered"}} if isinstance(battery, str) else battery
    report["thermal"] = shell("dumpsys", "thermalservice")
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
