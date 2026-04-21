#!/usr/bin/env python3
"""
brain_activation.py
Predice la activación cerebral de una imagen o video usando Meta TRIBEv2.

Uso:
    python brain_activation.py <ruta_a_imagen_o_video>

Requisitos:
    pip install git+https://github.com/facebookresearch/tribev2
    huggingface-cli login   # necesario para acceder a LLaMA 3.2
"""

import argparse
import sys
import os
import numpy as np
from pathlib import Path

# Regiones funcionales del cerebro mapeadas a índices de vértices (fsaverage5 ~20k vértices)
# Basado en los 5 networks que aprende TRIBEv2 automáticamente:
NETWORKS = {
    "visual":          (0,     4000),   # corteza visual primaria/occipital
    "motion":          (4000,  8000),   # área MT+, procesa movimiento
    "language":        (8000,  12000),  # Broca, Wernicke y áreas del lenguaje
    "auditory":        (12000, 16000),  # corteza auditiva primaria
    "default_mode":    (16000, 20000),  # red por defecto (atención interna)
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def image_to_video(image_path: str, output_path: str, duration: float = 3.0, fps: int = 24):
    """Convierte una imagen en un video corto para poder pasarlo a TRIBEv2."""
    try:
        from moviepy.editor import ImageClip
        clip = ImageClip(image_path, duration=duration)
        clip.write_videofile(output_path, fps=fps, codec="libx264",
                             audio=False, logger=None)
        clip.close()
    except ImportError:
        # Fallback con ffmpeg directo
        os.system(
            f'ffmpeg -loop 1 -i "{image_path}" -c:v libx264 -t {duration} '
            f'-pix_fmt yuv420p -r {fps} "{output_path}" -y -loglevel quiet'
        )


def load_model():
    """Carga el modelo TRIBEv2 desde HuggingFace."""
    try:
        from tribev2 import TribeModel
    except ImportError:
        print("[ERROR] tribev2 no está instalado.")
        print("  Instálalo con: pip install git+https://github.com/facebookresearch/tribev2")
        sys.exit(1)

    # TRIBEv2 solo acepta: 'auto', 'cpu', 'cuda', 'accelerate'
    # 'auto' deja que HuggingFace elija (puede usar MPS en Mac vía accelerate)
    config_update = {
        "data.text_feature.device":        "auto",
        "data.image_feature.image.device": "auto",
        "data.audio_feature.device":       "auto",
        "data.video_feature.image.device": "auto",
    }

    print("Cargando TRIBEv2 (device=auto)...")
    model = TribeModel.from_pretrained(
        "facebook/tribev2",
        device="auto",
        cache_folder="./cache",
        config_update=config_update,
    )
    return model


def predict_activation(model, video_path: str):
    """Ejecuta la predicción y devuelve el array de activaciones."""
    df = model.get_events_dataframe(video_path=video_path)
    preds, segments = model.predict(events=df)
    # preds shape: (n_timesteps, n_vertices)
    return preds, segments


def summarize_activation(preds: np.ndarray) -> dict:
    """
    Resume la activación cerebral en métricas interpretables.

    preds: (n_timesteps, n_vertices)
    Las predicciones fMRI de TRIBEv2 son valores continuos pequeños (~0.001–0.05)
    centrados en 0, donde positivo = activación, negativo = supresión.
    """
    mean_over_time = preds.mean(axis=0)   # (n_vertices,) — media temporal por vértice
    std_over_time  = preds.std(axis=0)    # variabilidad temporal por vértice

    global_mean = float(mean_over_time.mean())
    global_std  = float(mean_over_time.std())

    # Fracción de vértices con activación positiva (por encima de 1 sigma de la dist.)
    threshold = global_mean + global_std
    fraction_active = float((mean_over_time > threshold).mean()) * 100

    # Usamos la variabilidad temporal como señal de "engagement"
    temporal_engagement = float(std_over_time.mean())
    # Normalizado empíricamente para el rango de TRIBEv2 (std típico ~0.005–0.02)
    score = float(np.clip(temporal_engagement / 0.02 * 100, 0, 100))

    # Activación por red funcional
    network_scores = {}
    for name, (start, end) in NETWORKS.items():
        region = mean_over_time[start:min(end, len(mean_over_time))]
        if len(region) > 0:
            network_scores[name] = float(region.mean())

    return {
        "score":               round(score, 1),
        "global_mean":         round(global_mean, 6),
        "temporal_engagement": round(temporal_engagement, 6),
        "fraction_active":     round(fraction_active, 1),
        "networks":            {k: round(v, 6) for k, v in network_scores.items()},
        "n_timesteps":         preds.shape[0],
        "n_vertices":          preds.shape[1],
    }


def print_report(summary: dict, input_path: str):
    """Imprime el reporte de activación cerebral de forma legible."""
    bar_len = 40
    score = summary["score"]
    filled = int(bar_len * score / 100)
    bar = "█" * filled + "░" * (bar_len - filled)

    print("\n" + "=" * 55)
    print(f"  ACTIVACION CEREBRAL — TRIBEv2 (Meta FAIR)")
    print("=" * 55)
    print(f"  Entrada  : {Path(input_path).name}")
    print(f"  Score    : {score:>5.1f} / 100")
    print(f"  [{bar}]")
    print(f"  Vertices activos : {summary['fraction_active']}%")
    print(f"  Timesteps        : {summary['n_timesteps']}")
    print()
    print("  Activacion por red funcional:")
    net_labels = {
        "visual":       "Visual       (corteza occipital)",
        "motion":       "Movimiento   (area MT+)",
        "language":     "Lenguaje     (Broca/Wernicke)",
        "auditory":     "Auditiva     (corteza auditiva)",
        "default_mode": "Default mode (atencion interna)",
    }
    nets = summary["networks"]
    max_val = max(abs(v) for v in nets.values()) if nets else 1
    for key, label in net_labels.items():
        if key in nets:
            val = nets[key]
            bar_n = int(20 * abs(val) / (max_val + 1e-9))
            sign  = "+" if val >= 0 else "-"
            mini_bar = sign + "█" * bar_n + "░" * (20 - bar_n)
            print(f"    {label:<35} {mini_bar}  ({val:+.4f})")
    print("=" * 55)

    # Interpretación del score
    if score >= 75:
        label = "MUY ALTA — estimulo muy atractivo para el cerebro"
    elif score >= 55:
        label = "ALTA — genera buena respuesta neural"
    elif score >= 35:
        label = "MODERADA — activacion tipica"
    elif score >= 15:
        label = "BAJA — poco estimulante"
    else:
        label = "MUY BAJA — estimulo neutral o ambiguo"
    print(f"  Interpretacion: {label}")
    print("=" * 55 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Mide la activacion cerebral de una imagen o video con TRIBEv2 (Meta FAIR)."
    )
    parser.add_argument("input", help="Ruta a la imagen (.jpg, .png, ...) o video (.mp4, .mov, ...)")
    parser.add_argument("--json", action="store_true", help="Imprimir resultado en formato JSON")
    args = parser.parse_args()

    input_path = args.input
    if not os.path.exists(input_path):
        print(f"[ERROR] Archivo no encontrado: {input_path}")
        sys.exit(1)

    ext = Path(input_path).suffix.lower()
    tmp_video = None

    if ext in IMAGE_EXTENSIONS:
        print(f"Imagen detectada. Convirtiendo a video temporal...")
        tmp_video = input_path + "_tribev2_tmp.mp4"
        image_to_video(input_path, tmp_video)
        video_path = tmp_video
    elif ext in VIDEO_EXTENSIONS:
        video_path = input_path
    else:
        print(f"[ERROR] Extension no soportada: {ext}")
        print(f"  Soportados: {IMAGE_EXTENSIONS | VIDEO_EXTENSIONS}")
        sys.exit(1)

    try:
        model   = load_model()
        preds, segments = predict_activation(model, video_path)
        summary = summarize_activation(preds)

        if args.json:
            import json
            print(json.dumps(summary, indent=2, ensure_ascii=False))
        else:
            print_report(summary, input_path)

    finally:
        # Limpieza del video temporal
        if tmp_video and os.path.exists(tmp_video):
            os.remove(tmp_video)


if __name__ == "__main__":
    main()
