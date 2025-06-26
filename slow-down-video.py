#!/usr/bin/env python3
"""
slow-down-video.py – Ralentiza un video a 0.9× usando FFmpeg de forma segura.
"""

import sys
import subprocess
from pathlib import Path

FACTOR = 0.9

def slow_down_video(input_path: str, output_path: str, factor: float = FACTOR):
    inp = Path(input_path)
    out = Path(output_path)

    if not inp.exists():
        raise FileNotFoundError(f"Archivo no encontrado: {inp}")

    # FFmpeg admite encadenar atempo para valores fuera del rango
    atempo_filters = []
    f = factor
    while f < 0.5:
        atempo_filters.append("atempo=0.5")
        f /= 0.5
    atempo_filters.append(f"atempo={f}")
    af = ",".join(atempo_filters)

    cmd = [
        "ffmpeg", "-y", "-i", str(inp),
        "-filter:v", f"setpts={1/factor}*PTS",
        "-filter:a", af,
        "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart",
        str(out)
    ]

    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Uso: python slow-down-video.py <input.mp4> <output.mp4>")
        sys.exit(1)
    slow_down_video(*sys.argv[1:3])
