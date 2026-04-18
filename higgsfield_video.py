"""
higgsfield_video.py — Genera videos usando la API de Higgsfield AI.

Modos:
  image2video  Anima una imagen. Modelos: dop-standard, dop-preview, kling, seedance.

Uso:
  python higgsfield_video.py image2video --image foto.jpg --prompt "Camera slowly orbits"
  python higgsfield_video.py image2video --image foto.jpg --prompt "..." --model kling

Variables de entorno (.env):
  HIGGSFIELD_KEY_ID      — ID de la API key (ver cloud.higgsfield.ai/api-keys)
  HIGGSFIELD_KEY_SECRET  — Secreto de la API key
"""

import os
import sys
import time
import argparse
import requests
from dotenv import load_dotenv

load_dotenv()

# ── Configuración ──────────────────────────────────────────────────────────────

BASE_URL = "https://platform.higgsfield.ai"

KEY_ID     = os.getenv("HIGGSFIELD_KEY_ID")
KEY_SECRET = os.getenv("HIGGSFIELD_KEY_SECRET")

POLL_INTERVAL = 5    # segundos entre polling
MAX_WAIT      = 600  # segundos máximos de espera (10 min)

# ── Helpers ────────────────────────────────────────────────────────────────────

def auth_headers() -> dict:
    if not KEY_ID or not KEY_SECRET:
        print("ERROR: Falta HIGGSFIELD_KEY_ID o HIGGSFIELD_KEY_SECRET en .env")
        sys.exit(1)
    return {
        "Authorization": f"Key {KEY_ID}:{KEY_SECRET}",
        "Content-Type": "application/json",
    }


def upload_image(image_path: str) -> str:
    """Sube una imagen local a Higgsfield y devuelve la URL pública."""
    print(f"Subiendo imagen: {image_path}")

    # 1. Obtener URL pre-firmada
    import mimetypes
    content_type = mimetypes.guess_type(image_path)[0] or "image/png"

    resp = requests.post(
        f"{BASE_URL}/files/generate-upload-url",
        headers=auth_headers(),
        json={"content_type": content_type},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    upload_url = data["upload_url"]
    public_url = data["public_url"]

    # 2. Subir el archivo directamente (sin auth)
    with open(image_path, "rb") as f:
        put_resp = requests.put(
            upload_url,
            data=f,
            headers={"Content-Type": content_type},
            timeout=120,
        )
        put_resp.raise_for_status()

    print(f"Imagen subida: {public_url}")
    return public_url


def poll_status(request_id: str) -> dict:
    """Hace polling hasta que el job termine. Devuelve el resultado final."""
    url = f"{BASE_URL}/requests/{request_id}/status"
    elapsed = 0

    print(f"Esperando resultado (request_id={request_id})...")
    while elapsed < MAX_WAIT:
        resp = requests.get(url, headers=auth_headers(), timeout=30)
        resp.raise_for_status()
        data = resp.json()
        status = data.get("status", "").lower()

        print(f"  [{elapsed}s] Status: {status}")

        if status == "completed":
            return data
        if status in ("failed", "nsfw", "cancelled"):
            print(f"ERROR: Generación terminó con status '{status}'")
            print(data)
            sys.exit(1)

        time.sleep(POLL_INTERVAL)
        elapsed += POLL_INTERVAL

    print(f"ERROR: Tiempo de espera agotado ({MAX_WAIT}s)")
    sys.exit(1)


def download_video(video_url: str, output_path: str):
    """Descarga el video generado."""
    print(f"Descargando video desde: {video_url}")
    resp = requests.get(video_url, stream=True, timeout=120)
    resp.raise_for_status()
    with open(output_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"Video guardado en: {output_path}")


# ── Modos de generación ────────────────────────────────────────────────────────

def image_to_video(args):
    """Anima una imagen usando distintos modelos disponibles en Higgsfield."""

    # Mapeo de modelo a app path (URL: BASE_URL/{app_id})
    model_to_app = {
        "dop-standard": "higgsfield-ai/dop/standard",
        "dop-preview":  "higgsfield-ai/dop/preview",
        "kling":        "kling-video/v2.1/pro/image-to-video",
        "seedance":     "bytedance/seedance/v1/pro/image-to-video",
    }

    app_id = model_to_app.get(args.model)
    if not app_id:
        print(f"ERROR: Modelo desconocido '{args.model}'. Opciones: {list(model_to_app)}")
        sys.exit(1)

    # Resolver imagen: URL o archivo local
    if args.image.startswith("http://") or args.image.startswith("https://"):
        image_url = args.image
    else:
        image_url = upload_image(args.image)

    payload = {
        "prompt":    args.prompt,
        "image_url": image_url,
    }
    # Parámetros específicos de DoP
    if args.model.startswith("dop"):
        payload["motion_strength"] = args.motion_strength
        payload["enhance_prompt"]  = not args.no_enhance
        payload["check_nsfw"]      = True
        if args.motion_id:
            payload["motion_id"] = args.motion_id
    if args.seed is not None:
        payload["seed"] = args.seed

    print(f"Enviando job image-to-video (model={args.model}, app={app_id})...")
    resp = requests.post(
        f"{BASE_URL}/{app_id}",
        headers=auth_headers(),
        json=payload,
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()

    request_id = data.get("request_id")
    if not request_id:
        print("ERROR: No se recibió request_id:", data)
        sys.exit(1)

    result = poll_status(request_id)

    video_url = result.get("video", {}).get("url")
    if not video_url:
        print("ERROR: No hay URL de video en la respuesta:", result)
        sys.exit(1)

    download_video(video_url, args.output)


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Genera videos con la API de Higgsfield AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    # ── image2video ──
    p_i2v = subparsers.add_parser("image2video", help="Anima una imagen")
    p_i2v.add_argument("--image",           required=True, help="Ruta local o URL de la imagen")
    p_i2v.add_argument("--prompt",          required=True, help="Descripción del movimiento de cámara")
    p_i2v.add_argument("--model",           default="dop-standard",
                        choices=["dop-standard", "dop-preview", "kling", "seedance"],
                        help="Modelo a usar (default: dop-standard)")
    p_i2v.add_argument("--motion-id",       default=None,
                        help="UUID del motion preset, solo para modelos DoP (opcional)")
    p_i2v.add_argument("--motion-strength", type=float, default=0.8,
                        help="Intensidad del movimiento 0.0–1.0, solo DoP (default: 0.8)")
    p_i2v.add_argument("--seed",            type=int, default=None)
    p_i2v.add_argument("--no-enhance",      action="store_true",
                        help="Desactivar enhance_prompt (solo DoP)")
    p_i2v.add_argument("--output",          default="higgsfield_output.mp4",
                        help="Archivo de salida (default: higgsfield_output.mp4)")

    args = parser.parse_args()

    if args.mode == "image2video":
        image_to_video(args)


if __name__ == "__main__":
    main()
