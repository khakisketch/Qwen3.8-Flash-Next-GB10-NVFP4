#!/usr/bin/env python3
"""
GB10 (DGX Spark) hardware/environment doctor.

Collects and validates the host facts required to run a Qwen3.8-Flash-Next
GB10 deployment, then writes a JSON report suitable for the model card.

No third-party dependencies required; optional PyTorch capability check is
included when torch is importable.
"""

import argparse
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone

REQUIRED_ARCH = {"aarch64", "arm64"}


def run(cmd, **kwargs):
    try:
        out = subprocess.run(
            cmd, capture_output=True, text=True, timeout=kwargs.pop("timeout", 30),
            **kwargs,
        )
        return out.returncode, (out.stdout or "").strip(), (out.stderr or "").strip()
    except FileNotFoundError:
        return None, "", ""
    except subprocess.TimeoutExpired:
        return -1, "", "timeout"


def first_line(rc, out, err):
    return (out or err).splitlines()[0] if (out or err) else ""


def query_nvidia_smi():
    report = {}
    rc, out, err = run([
        "nvidia-smi",
        "--query-gpu=name,driver_version,compute_cap,vbios_version,power.draw,power.limit,clocks.sm,clocks.max.sm",
        "--format=csv,noheader,nounits",
    ])
    report["present"] = rc == 0
    report["raw"] = out if rc == 0 else err
    if rc == 0 and out:
        fields = ["name", "driver_version", "compute_cap", "vbios", "power_draw_w",
                  "power_limit_w", "sm_clk_mhz", "sm_clk_max_mhz"]
        vals = out.splitlines()[0].split(", ")
        report["detail"] = dict(zip(fields, vals))
    # cuda driver version
    rc, out, err = run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
    report["driver_version"] = first_line(rc, out, err)
    rc, out, err = run(["nvidia-smi"])
    report["cuda_version"] = None
    for line in (out or "").splitlines():
        if "CUDA Version" in line:
            report["cuda_version"] = line.split("CUDA Version:")[-1].strip().split()[0]
    return report


def query_cpu():
    report = {}
    report["machine"] = platform.machine()
    report["arch_ok"] = platform.machine().lower() in REQUIRED_ARCH
    rc, out, err = run(["lscpu"])
    for line in (out or "").splitlines():
        low = line.lower()
        if "model name" in low:
            report["model"] = line.split(":", 1)[-1].strip()
        if line.startswith("CPU(s)"):
            report["cpus"] = line.split(":", 1)[-1].strip()
    return report


def query_memory():
    report = {}
    rc, out, err = run(["free", "-h"])
    report["free_raw"] = out
    report["mem_total_kb"] = None
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal"):
                    report["mem_total_kb"] = int(line.split()[1])
                if line.startswith("MemAvailable"):
                    report["mem_available_kb"] = int(line.split()[1])
                if line.startswith("SwapTotal"):
                    report["swap_total_kb"] = int(line.split()[1])
    except OSError:
        pass
    return report


def query_storage():
    report = {}
    rc, out, err = run(["df", "-h", "/"])
    report["root_disk"] = out
    rc, out, err = run(["df", "-B1", "--output=source,size,avail,fstype,target", "/"])
    report["root_bytes"] = out
    report["nvme"] = []
    rc, out, err = run(["ls", "-1", "/dev/nvme*n1"])
    if rc == 0:
        report["nvme"] = [l for l in out.splitlines() if l]
    return report


def query_docker():
    report = {}
    rc, out, err = run(["docker", "--version"])
    report["docker"] = first_line(rc, out, err)
    rc, out, err = run(["docker", "info", "--format", "{{.ServerVersion}}"])
    report["docker_server"] = first_line(rc, out, err)
    rc, out, err = run(["nvidia-ctk", "--version"])
    report["nvidia_ctk"] = first_line(rc, out, err)
    # quick GPU-mount smoke test
    rc, out, err = run([
        "docker", "run", "--rm", "--gpus", "all",
        "nvcr.io/nvidia/cuda:12.9.0-base-ubuntu22.04", "nvidia-smi", "-L",
    ], timeout=120)
    report["gpu_container_ok"] = rc == 0
    report["gpu_container_out"] = first_line(rc, out, err)
    return report


def query_torch():
    report = {"available": False}
    try:
        import torch
        report["available"] = torch.cuda.is_available()
        report["torch_version"] = torch.__version__
        if torch.cuda.is_available():
            report["device_name"] = torch.cuda.get_device_name(0)
            report["capability"] = list(torch.cuda.get_device_capability(0))
            report["mem_total_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
    except Exception as e:  # noqa: BLE001
        report["error"] = str(e)
    return report


def main():
    ap = argparse.ArgumentParser(description="GB10 doctor")
    ap.add_argument("--output", default="gb10_doctor.json")
    ap.add_argument("--skip-docker", action="store_true", help="skip docker smoke test")
    args = ap.parse_args()

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "python": platform.python_version(),
        "cpu": query_cpu(),
        "gpu": query_nvidia_smi(),
        "memory": query_memory(),
        "storage": query_storage(),
    }
    if not args.skip_docker:
        report["docker"] = query_docker()
    report["torch"] = query_torch()

    # convenience summary
    arch_ok = report["cpu"]["arch_ok"]
    gpu_ok = report["gpu"].get("present") and "GB10" in (report["gpu"].get("raw") or "")
    mem_gb = (report["memory"].get("mem_total_kb") or 0) / 1e6
    report["summary"] = {
        "arch_ok": arch_ok,
        "gb10_detected": bool(gpu_ok),
        "ram_gb": round(mem_gb, 1),
        "docker_gpu_backend": report.get("docker", {}).get("gpu_container_ok", None),
    }

    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report["summary"], indent=2))
    print(f"\nFull report written to: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())