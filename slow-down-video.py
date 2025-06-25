#!/usr/bin/env python3
"""
slow-down-video.py  – Reduce la velocidad de un vídeo a 0.9×.

Uso:
    python slow-down-video.py input.mp4 output.mp4     # genera output.mp4 a 0.9×
"""

import sys
import subprocess
from pathlib import Path
FACTOR = 0.9  # 0.9×

def slow_ffmpeg(input_path: str, output_path: str, factor: float = FACTOR):
    inp = Path(input_path)
    out = Path(output_path)
    if not inp.exists():
        raise FileNotFoundError(inp)

    vf = f"setpts=1/{factor}*PTS"
    af = f"atempo={factor}"           # o rubberband=tempo=0.9:pitch=1.0
    cmd = [
        "ffmpeg", "-y", "-i", str(inp),
        "-filter_complex", f"[0:v]{vf}[v];[0:a]{af}[a]",
        "-map", "[v]", "-map", "[a]",
        "-c:v", "libx264", "-crf", "19", "-preset", "medium",
        "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart",
        str(out)
    ]
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Uso: python slow-down-video.py <input.mp4> <output.mp4>")
        sys.exit(1)
    slow_ffmpeg(*sys.argv[1:3])