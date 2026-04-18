"""
seedance_video.py — Genera videos con Seedance 1.5 Pro via Replicate (sin markup de Higgsfield).

Uso:
  # image-to-video
  python seedance_video.py --image foto.jpg --prompt "Slow cinematic dolly forward"

  # text-to-video
  python seedance_video.py --prompt "A majestic eagle soaring over mountains at sunset"

  # Opciones extra
  python seedance_video.py --image foto.jpg --prompt "..." --duration 10 --aspect-ratio 9:16 --output out.mp4

Variables de entorno (.env):
  REPLICATE_API_TOKEN  — token de Replicate (replicate.com/account/api-tokens)
"""

import os
import sys
import time
import argparse
import requests
from dotenv import load_dotenv

load_dotenv()

# ── Configuración ──────────────────────────────────────────────────────────────

REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")
MODEL               = "bytedance/seedance-1.5-pro"
BASE_URL            = "https://api.replicate.com/v1"

POLL_INTERVAL = 5
MAX_WAIT      = 600

VALID_ASPECT_RATIOS = ["16:9", "9:16", "1:1", "4:3", "3:4", "21:9"]
VALID_FPS           = [24, 30]

# ── Helpers ────────────────────────────────────────────────────────────────────

def auth_headers() -> dict:
    if not REPLICATE_API_TOKEN:
        print("ERROR: Falta REPLICATE_API_TOKEN en .env")
        sys.exit(1)
    return {
        "Authorization": f"Bearer {REPLICATE_API_TOKEN}",
        "Content-Type": "application/json",
    }


def encode_image_as_data_uri(file_path: str) -> str:
    """Codifica una imagen local como data URI base64."""
    import base64, mimetypes
    content_type = mimetypes.guess_type(file_path)[0] or "image/png"
    with open(file_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{content_type};base64,{b64}"


def poll_prediction(prediction_id: str) -> dict:
    """Polling hasta que la predicción termine."""
    url = f"{BASE_URL}/predictions/{prediction_id}"
    elapsed = 0
    print(f"Esperando resultado (id={prediction_id})...")

    while elapsed < MAX_WAIT:
        resp = requests.get(url, headers=auth_headers(), timeout=30)
        resp.raise_for_status()
        data = resp.json()
        status = data.get("status", "")

        print(f"  [{elapsed}s] Status: {status}")

        if status == "succeeded":
            return data
        if status in ("failed", "canceled"):
            print(f"ERROR: Predicción terminó con status '{status}'")
            print(data.get("error"))
            sys.exit(1)

        time.sleep(POLL_INTERVAL)
        elapsed += POLL_INTERVAL

    print(f"ERROR: Tiempo de espera agotado ({MAX_WAIT}s)")
    sys.exit(1)


def download_video(video_url: str, output_path: str):
    """Descarga el video generado."""
    print(f"Descargando video: {video_url}")
    resp = requests.get(video_url, stream=True, timeout=120)
    resp.raise_for_status()
    with open(output_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"Video guardado en: {output_path}")


# ── Generación ─────────────────────────────────────────────────────────────────

def generate(args):
    input_payload = {
        "prompt":         args.prompt,
        "duration":       args.duration,
        "fps":            args.fps,
        "camera_fixed":   args.camera_fixed,
        "generate_audio": args.generate_audio,
    }

    if args.image:
        if args.image.startswith("http://") or args.image.startswith("https://"):
            input_payload["image"] = args.image
        else:
            input_payload["image"] = encode_image_as_data_uri(args.image)
        # aspect_ratio se ignora cuando hay imagen según la doc
    else:
        input_payload["aspect_ratio"] = args.aspect_ratio

    if args.last_frame:
        if args.last_frame.startswith("http://") or args.last_frame.startswith("https://"):
            input_payload["last_frame_image"] = args.last_frame
        else:
            input_payload["last_frame_image"] = encode_image_as_data_uri(args.last_frame)

    if args.seed is not None:
        input_payload["seed"] = args.seed

    print(f"Enviando job a Replicate ({MODEL})...")
    resp = requests.post(
        f"{BASE_URL}/models/{MODEL}/predictions",
        headers=auth_headers(),
        json={"input": input_payload},
        timeout=60,
    )
    resp.raise_for_status()
    prediction = resp.json()
    prediction_id = prediction.get("id")
    if not prediction_id:
        print("ERROR: No se recibió prediction id:", prediction)
        sys.exit(1)

    result = poll_prediction(prediction_id)

    output = result.get("output")
    if not output:
        print("ERROR: No hay output en la respuesta:", result)
        sys.exit(1)

    # output puede ser string o lista
    video_url = output if isinstance(output, str) else output[0]
    download_video(video_url, args.output)


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Genera videos con Seedance 1.5 Pro via Replicate",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument("--prompt",          required=True,
                        help="Descripción del video o movimiento de cámara")
    parser.add_argument("--image",           default=None,
                        help="Ruta local o URL de la imagen inicial (omitir = text-to-video)")
    parser.add_argument("--last-frame",      default=None,
                        help="Ruta local o URL de la imagen final (solo si --image está presente)")
    parser.add_argument("--duration",        type=int, default=5, metavar="SECS",
                        help="Duración en segundos, 2–12 (default: 5)")
    parser.add_argument("--aspect-ratio",    default="16:9", choices=VALID_ASPECT_RATIOS,
                        help="Aspect ratio (ignorado si se usa --image). Default: 16:9")
    parser.add_argument("--fps",             type=int, default=24, choices=VALID_FPS,
                        help="Frames por segundo (default: 24)")
    parser.add_argument("--camera-fixed",    action="store_true",
                        help="Fijar la cámara (sin movimiento)")
    parser.add_argument("--generate-audio",  action="store_true",
                        help="Generar audio sincronizado con el video")
    parser.add_argument("--seed",            type=int, default=None,
                        help="Semilla para reproducibilidad")
    parser.add_argument("--output",          default="seedance_output.mp4",
                        help="Archivo de salida (default: seedance_output.mp4)")

    args = parser.parse_args()

    if not (2 <= args.duration <= 12):
        print("ERROR: --duration debe estar entre 2 y 12 segundos")
        sys.exit(1)

    if args.last_frame and not args.image:
        print("ERROR: --last-frame requiere también --image")
        sys.exit(1)

    generate(args)


if __name__ == "__main__":
    main()
