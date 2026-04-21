import os
import re
import json
import uuid
import random
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from huggingface_hub import login
from PIL import Image

if not hasattr(Image, "ANTIALIAS"):
    Image.ANTIALIAS = Image.Resampling.LANCZOS

from moviepy.editor import *
import requests
import time
from moviepy.editor import TextClip
from elevenlabs.client import ElevenLabs
import traceback
from moviepy.editor import VideoFileClip
import argparse
from requests.exceptions import Timeout

# === CONFIGURACIÓN ===
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
login(token=HF_TOKEN)
DEEPSEEK_API_KEY    = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_ENDPOINT   = "https://api.deepseek.com/v1/chat/completions"
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")
# IDEAS_FILE = "ideas.json"
# IDEAS_FILE = "real-stories.json"
# IDEAS_FILE = "argentina-football-players.json"
IDEAS_FILE = None  # Se asignará por argparse más abajo

OUTPUT_DIR = "stories"
os.environ["IMAGEMAGICK_BINARY"] = "/opt/homebrew/bin/convert"  # o el path que te dé `which convert`
FINAL_WIDTH = 1080
FINAL_HEIGHT = 1920
SILENCIO_SEGUNDOS = 0.5
MAX_REINTENTOS = 10
MODO_ANIMADO = True  # Cambiar a False para usar imágenes estáticas
SHOULD_INCLUDE_SUBTITLES = True

# ================================================================
# PROVIDER CONFIGURATION — cambiar estos valores para switching
# ================================================================

IMAGE_PROVIDER   = "hf-default"   # "flux" | "hf-default"
HF_IMAGE_MODEL   = "black-forest-labs/FLUX.1-schnell"
HF_IMAGE_WIDTH   = 768            # solo para IMAGE_PROVIDER = "flux"
HF_IMAGE_HEIGHT  = 1344           # solo para IMAGE_PROVIDER = "flux"

ELEVENLABS_VOICE_ID = "80lPKtzJMPh1vjYMUgwe"
ELEVENLABS_MODEL_ID = "eleven_multilingual_v2"

DEEPSEEK_MODEL = "deepseek-chat"

REPLICATE_SEEDANCE_MODEL      = "bytedance/seedance-2.0-fast"  # "bytedance/seedance-1.5-pro" | "bytedance/seedance-2.0-fast"
REPLICATE_SEEDANCE_RESOLUTION = "480p"   # "480p" | "720p" | "1080p"
REPLICATE_SEEDANCE_MODE       = "t2v"    # "i2v" (imagen→video) | "t2v" (texto→video)


def sanitize_filename(text):
    return re.sub(r'[^a-zA-Z0-9_-]', '_', text).lower()

def elegir_idea():
    with open(IDEAS_FILE, "r", encoding="utf-8") as f:
        ideas = json.load(f)
    return random.choice(ideas)

def generar_prompt(idea):
    return f"""
Quiero que generes un video corto estilo "reel" de 2 minutos, basado en la siguiente historia real sobre una figura histórica célebre:

{idea['descripcion']}

El objetivo es emocionar, sorprender e informar al espectador, manteniendo su atención hasta el final. No debe ser una biografía plana, sino una narrativa poderosa basada en hechos reales, que destaque un momento clave, un dilema, un conflicto o una decisión crucial en la vida de ese personaje.

El video debe tener estos elementos:

1. `textos`: una lista ordenada de fragmentos narrativos para voz en off. Cada uno con duración (`milisegundos`) y contenido (`texto`). 
   - El primer fragmento debe captar la atención con una pregunta intrigante, una afirmación inesperada o una situación límite.
   - El relato debe tener introducción (contexto histórico breve), desarrollo (tensión, dilema, desafío) y un cierre emocional o inspirador.
   - Mostrá claramente la curva emocional.
   - IMPORTANTE: El texto debe ser exactamente lo narrado, no incluyas aclaraciones (como indicar silencios, o sonidos de fondo, etc) ya que se leerán y no tendrian sentido al escucharse

2. `imagenes`: una lista con descripciones visuales alineadas a cada texto, también con `milisegundos` y `descripcion`.
   - Indicá tipo de plano (general, primer plano, detalle).
   - Todas deben compartir un estilo visual coherente y representativo de la época histórica.

3. `audio`: nombre exacto de una pieza instrumental real (sin letra), disponible en YouTube, que intensifique el tono narrativo del video. Puede ser épico, melancólico, intrigante o inspirador según el caso.

⚠️ Agregá un campo `contexto_visual_global` con detalles sobre:
- Estética cinematográfica (películas, series o documentales que inspiren el estilo visual)
- Paleta de colores
- Iluminación, clima y época histórica representada

Formato de salida:
{{"textos":[{{"milisegundos":0,"texto":"..."}}], "imagenes":[{{"milisegundos":0,"descripcion":"..."}}], "audio":"...", "contexto_visual_global": "..."}}.
"""

def _llm_deepseek(prompt):
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": [
            {"role": "system", "content": "Sos un guionista experto en reels"},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.8
    }
    response = requests.post(DEEPSEEK_ENDPOINT, headers=headers, json=payload)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]

def llamar_llm(prompt):
    return _llm_deepseek(prompt)

def extraer_json(respuesta):
    inicio = respuesta.find("{")
    fin = respuesta.rfind("}") + 1
    return json.loads(respuesta[inicio:fin])

def guardar_historia_json(story_dir, idea, contenido_json):
    json_path = os.path.join(story_dir, "story.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "titulo": idea["titulo"],
            "descripcion": idea["descripcion"],
            **contenido_json
        }, f, ensure_ascii=False, indent=2)
    return json_path

def ralentizar_video(input_path, duracion_objetivo, output_path):
    clip = VideoFileClip(input_path)
    if duracion_objetivo <= 0 or clip.duration <= 0:
        clip.close()
        import shutil; shutil.copy(input_path, output_path)
        return
    velocidad = clip.duration / duracion_objetivo
    if velocidad > 0:
        clip_ralentizado = clip.fx(vfx.speedx, factor=velocidad)
        clip_ralentizado = clip_ralentizado.set_duration(duracion_objetivo)
        clip_ralentizado.write_videofile(output_path, codec="libx264", audio=False, fps=24)
        clip_ralentizado.close()
    clip.close()
    
SEEDANCE_DURATION_MIN = 2
SEEDANCE_DURATION_MAX = 12
REPLICATE_BASE = "https://api.replicate.com/v1"

def _build_replicate_input(img_path, prompt, duracion):
    import base64, mimetypes
    duracion_api = max(SEEDANCE_DURATION_MIN, min(SEEDANCE_DURATION_MAX, round(duracion)))
    payload = {"prompt": prompt, "duration": duracion_api}
    if REPLICATE_SEEDANCE_MODE == "i2v":
        content_type = mimetypes.guess_type(img_path)[0] or "image/png"
        with open(img_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        payload["image"] = f"data:{content_type};base64,{b64}"
    else:
        payload["aspect_ratio"] = "9:16"
    payload["resolution"] = REPLICATE_SEEDANCE_RESOLUTION
    payload["generate_audio"] = False
    return payload, duracion_api


def _submit_replicate_prediction(input_payload):
    headers = {"Authorization": f"Bearer {REPLICATE_API_TOKEN}", "Content-Type": "application/json"}
    resp = requests.post(
        f"{REPLICATE_BASE}/models/{REPLICATE_SEEDANCE_MODEL}/predictions",
        headers=headers, json={"input": input_payload}, timeout=60,
    )
    resp.raise_for_status()
    prediction_id = resp.json().get("id")
    if not prediction_id:
        raise ValueError(f"Replicate no devolvió prediction id: {resp.json()}")
    return prediction_id


def _poll_replicate_predictions(jobs):
    """Polling de múltiples predictions en paralelo. jobs: {idx: prediction_id}
    Retorna {idx: result_data} — los fallidos incluyen clave 'error'."""
    headers = {"Authorization": f"Bearer {REPLICATE_API_TOKEN}"}
    pending = dict(jobs)
    results = {}
    elapsed = 0
    while pending and elapsed < 600:
        for idx in list(pending.keys()):
            data = requests.get(f"{REPLICATE_BASE}/predictions/{pending[idx]}", headers=headers, timeout=30).json()
            status = data.get("status", "")
            if status == "succeeded":
                results[idx] = data
                del pending[idx]
                print(f"  [{elapsed}s] ✅ Clip {idx:03d} listo")
            elif status in ("failed", "canceled"):
                results[idx] = {"error": data.get("error"), "status": status}
                del pending[idx]
                print(f"  [{elapsed}s] ❌ Clip {idx:03d} falló: {data.get('error')}")
        if pending:
            print(f"  [{elapsed}s] ⏳ Esperando {len(pending)} clips: {list(pending.keys())}")
            time.sleep(5)
            elapsed += 5
    return results


def _download_replicate_result(result, output_path, duracion):
    output = result.get("output")
    if not output:
        raise ValueError(f"No hay output en la respuesta: {result}")
    video_url = output if isinstance(output, str) else output[0]
    r = requests.get(video_url, stream=True, timeout=120)
    r.raise_for_status()
    duracion_api = max(SEEDANCE_DURATION_MIN, min(SEEDANCE_DURATION_MAX, round(duracion)))
    if abs(duracion_api - duracion) > 0.1:
        temp_path = output_path.replace(".mp4", "_seedance_raw.mp4")
        with open(temp_path, "wb") as f:
            for chunk in r.iter_content(8192):
                f.write(chunk)
        ralentizar_video(temp_path, duracion, output_path)
    else:
        with open(output_path, "wb") as f:
            for chunk in r.iter_content(8192):
                f.write(chunk)


def _build_hf_image_client():
    if IMAGE_PROVIDER == "flux":
        return InferenceClient(model=HF_IMAGE_MODEL, token=HF_TOKEN, provider="black-forest-labs")
    return InferenceClient(model=HF_IMAGE_MODEL, token=HF_TOKEN)

def _generate_single_image(hf_client, prompt):
    if IMAGE_PROVIDER == "flux":
        return hf_client.text_to_image(
            prompt,
            width=HF_IMAGE_WIDTH,
            height=HF_IMAGE_HEIGHT,
            num_inference_steps=28,
            guidance_scale=3.5,
        )
    return hf_client.text_to_image(prompt)

def _calcular_duraciones_imagenes(imagenes, audio_dir=None, duracion_default=5.0):
    """Devuelve duraciones por imagen.

    Si audio_dir existe y tiene los fragmentos .mp3, usa esas duraciones (fuente de verdad).
    Sino, calcula la diferencia entre timestamps consecutivos del JSON como fallback.
    """
    n = len(imagenes)

    if audio_dir and os.path.isdir(audio_dir):
        fragmentos = sorted([
            f for f in os.listdir(audio_dir)
            if f.endswith(".mp3") and not f.startswith("cuento") and not f.startswith("silencio")
        ])
        if len(fragmentos) == n:
            duraciones = []
            for i, fname in enumerate(fragmentos):
                clip = AudioFileClip(os.path.join(audio_dir, fname))
                dur = clip.duration
                clip.close()
                if i < n - 1:
                    dur += SILENCIO_SEGUNDOS
                duraciones.append(dur)
            print(f"📐 Duraciones calculadas desde audios: {[f'{d:.1f}s' for d in duraciones]}")
            return duraciones

    # Fallback: diferencia de timestamps del JSON
    duraciones = []
    for i, img in enumerate(imagenes):
        ms_actual = img.get("milisegundos", 0)
        if i + 1 < n:
            ms_siguiente = imagenes[i + 1].get("milisegundos", ms_actual)
            dur = (ms_siguiente - ms_actual) / 1000.0
        else:
            dur = duracion_default
        duraciones.append(dur if dur > 0 else duracion_default)
    return duraciones


def _static_video_fallback(img_path, anim_path, duracion):
    clip = ImageClip(img_path, duration=duracion).resize(height=FINAL_HEIGHT)
    clip.write_videofile(anim_path, fps=24, codec="libx264", audio=False, logger=None)
    clip.close()


def generar_imagenes(imagenes, image_dir, contexto_visual_global=None, max_reintentos=10, audio_dir=None):
    os.makedirs(image_dir, exist_ok=True)
    hf_client = _build_hf_image_client()
    duraciones = _calcular_duraciones_imagenes(imagenes, audio_dir=audio_dir)

    # ── Fase 1: generar todas las imágenes en paralelo ────────────────────────
    def _gen_image(idx, img):
        prompt = f"Imagen a dibujar: {img['descripcion']}\n\nContexto: {contexto_visual_global}." if contexto_visual_global else img["descripcion"]
        for intento in range(max_reintentos):
            try:
                print(f"🖼️ Generando imagen {idx:03} (intento {intento+1})")
                image = _generate_single_image(hf_client, prompt)
                img_path = os.path.join(image_dir, f"{idx:03}.png")
                image.save(img_path)
                return idx, img_path, prompt
            except Timeout:
                print(f"⚠️ Timeout imagen {idx:03}")
            except Exception as e:
                print(f"❌ Error imagen {idx:03}: {e}")
                if intento < max_reintentos - 1:
                    time.sleep(2)
        print(f"🚫 No se pudo generar imagen {idx:03}")
        return idx, None, None

    img_results = {}
    with ThreadPoolExecutor(max_workers=4) as ex:
        for idx, img_path, prompt in ex.map(lambda a: _gen_image(*a), enumerate(imagenes, 1)):
            if img_path:
                img_results[idx] = (img_path, prompt)

    if not MODO_ANIMADO:
        return

    # ── Fase 2: enviar todos los jobs a Replicate ────────────────────────────
    submitted = {}  # {idx: (prediction_id, duracion, img_path)}
    for idx, img in enumerate(imagenes, 1):
        if idx not in img_results:
            continue
        img_path, img_prompt = img_results[idx]
        duracion = duraciones[idx - 1]
        video_prompt = (f"{img['descripcion']}. {contexto_visual_global}" if contexto_visual_global else img["descripcion"]) if REPLICATE_SEEDANCE_MODE == "t2v" else img_prompt
        try:
            input_payload, _ = _build_replicate_input(img_path, video_prompt, duracion)
            pid = _submit_replicate_prediction(input_payload)
            submitted[idx] = (pid, duracion, img_path)
            print(f"🚀 Job {idx:03} enviado (id={pid[:8]}...)")
        except Exception as e:
            print(f"⚠️ No se pudo enviar job {idx:03}: {e}. Fallback estático.")
            _static_video_fallback(img_path, os.path.join(image_dir, f"{idx:03}.mp4"), duracion)

    if not submitted:
        return

    # ── Fase 3: polling de todos los jobs juntos ─────────────────────────────
    print(f"\n⏳ Esperando {len(submitted)} clips de Replicate en paralelo...")
    poll_results = _poll_replicate_predictions({idx: pid for idx, (pid, _, _) in submitted.items()})

    # ── Fase 4: descargar todos los videos ───────────────────────────────────
    for idx, (pid, duracion, img_path) in submitted.items():
        anim_path = os.path.join(image_dir, f"{idx:03}.mp4")
        result = poll_results.get(idx, {})
        if result.get("error"):
            print(f"⚠️ Clip {idx:03} falló ({result['error']}). Fallback estático.")
            _static_video_fallback(img_path, anim_path, duracion)
        else:
            try:
                print(f"⬇️  Descargando clip {idx:03}...")
                _download_replicate_result(result, anim_path, duracion)
                print(f"✅ Clip {idx:03} guardado")
            except Exception as e:
                print(f"⚠️ Error descargando clip {idx:03}: {e}. Fallback estático.")
                _static_video_fallback(img_path, anim_path, duracion)





def _tts_elevenlabs(text, filename, elevenlabs_client):
    audio = elevenlabs_client.text_to_speech.convert(
        voice_id=ELEVENLABS_VOICE_ID,
        model_id=ELEVENLABS_MODEL_ID,
        text=text,
    )
    with open(filename, "wb") as f:
        for chunk in audio:
            f.write(chunk)

def generar_audios(textos, audio_dir):
    os.makedirs(audio_dir, exist_ok=True)

    elevenlabs_client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))

    silencio_path = os.path.join(audio_dir, "silencio.mp3")
    if not os.path.exists(silencio_path):
        subprocess.run([
            "ffmpeg", "-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono",
            "-t", str(SILENCIO_SEGUNDOS), "-q:a", "9", "-acodec", "libmp3lame", silencio_path
        ])

    def _gen_audio(idx, fragmento):
        texto = fragmento["texto"]
        filename = os.path.abspath(os.path.join(audio_dir, f"{idx:03}.mp3"))
        for attempt in range(5):
            try:
                _tts_elevenlabs(texto, filename, elevenlabs_client)
                break
            except Exception as e:
                if "429" in str(e) or "concurrent_limit" in str(e):
                    wait = 5 * (attempt + 1)
                    print(f"⏳ Rate limit TTS {idx:03}, reintentando en {wait}s...")
                    time.sleep(wait)
                else:
                    raise
        clip = AudioFileClip(filename)
        dur = clip.duration
        clip.close()
        print(f"🎙️ Fragmento {idx:03} generado ({dur:.2f}s): {texto[:40]}...")
        return idx, filename, dur

    results = {}
    with ThreadPoolExecutor(max_workers=3) as ex:
        futures = {ex.submit(_gen_audio, idx, frag): idx for idx, frag in enumerate(textos, 1)}
        for future in as_completed(futures):
            idx, filename, dur = future.result()
            results[idx] = (filename, dur)

    audio_files = [results[i][0] for i in range(1, len(textos) + 1)]
    durations   = [results[i][1] for i in range(1, len(textos) + 1)]

    concat_path = os.path.join(audio_dir, "concat_list.txt")
    with open(concat_path, "w", encoding="utf-8") as f:
        for i, file in enumerate(audio_files):
            rel_path = os.path.relpath(file, audio_dir)
            f.write(f"file '{rel_path}'\n")
            if i != len(audio_files) - 1:
                f.write(f"file 'silencio.mp3'\n")

    final_audio = os.path.join(audio_dir, "cuento_completo.mp3")
    subprocess.run([
        "ffmpeg", "-f", "concat", "-safe", "0", "-i", "concat_list.txt",
        "-c", "copy", "cuento_completo.mp3"
    ], cwd=audio_dir)
    return final_audio, durations

def unir_audios_fragmentados(audio_dir, num_fragmentos):
    silencio_path = os.path.join(audio_dir, "silencio.mp3")
    if not os.path.exists(silencio_path):
        subprocess.run([
            "ffmpeg", "-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono",
            "-t", str(SILENCIO_SEGUNDOS), "-q:a", "9", "-acodec", "libmp3lame", "silencio.mp3"
        ], cwd=audio_dir)

    audio_files = [f"{i:03}.mp3" for i in range(1, num_fragmentos + 1)]
    concat_path = os.path.join(audio_dir, "concat_list.txt")
    with open(concat_path, "w", encoding="utf-8") as f:
        for i, filename in enumerate(audio_files):
            f.write(f"file '{filename}'\n")
            if i != len(audio_files) - 1:
                f.write(f"file 'silencio.mp3'\n")

    subprocess.run([
        "ffmpeg", "-f", "concat", "-safe", "0", "-i", "concat_list.txt",
        "-c", "copy", "cuento_completo.mp3"
    ], cwd=audio_dir)
    print(f"✅ Audio final generado: {os.path.join(audio_dir, 'cuento_completo.mp3')}")


def _download_music_local():
    music_dir = os.path.join("assets", "music")
    if not os.path.exists(music_dir):
        return None
    archivos = [f for f in os.listdir(music_dir) if f.endswith(".mp3")]
    if not archivos:
        return None
    return os.path.join(music_dir, random.choice(archivos))

def download_music():
    return _download_music_local()

def generar_video(textos, duraciones, image_dir, narracion_path, musica_path, output_path):
    print(f"\n🎬 Ensamblando video — {len(duraciones)} clips, {sum(duraciones):.1f}s total")
    clips = []

    for idx, dur in enumerate(duraciones, 1):
        if MODO_ANIMADO:
            video_path = os.path.join(image_dir, f"{idx:03}.mp4")
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Clip {idx:03}.mp4 no encontrado en {image_dir}")
            vc = VideoFileClip(video_path)
            print(f"  🎞️  Clip {idx:03} — archivo: {video_path} | duración archivo: {vc.duration:.2f}s | duración objetivo: {dur:.2f}s")
            base_clip = vc.subclip(0, min(dur, vc.duration)).resize(height=1920)
        else:
            img_path = os.path.join(image_dir, f"{idx:03}.png")
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Imagen {idx:03}.png no encontrada en {image_dir}")
            print(f"  🖼️  Clip {idx:03} — imagen estática | duración: {dur:.2f}s")
            base_clip = ImageClip(img_path, duration=dur).resize(height=1920).fadein(0.2).fadeout(0.2)

        zoom_type = "in"
        zoom_factor_start = 1.0 if zoom_type == "in" else 1.1
        zoom_factor_end = 1.1 if zoom_type == "in" else 1.0

        texto_subtitulo = textos[idx - 1]["texto"]

        if SHOULD_INCLUDE_SUBTITLES:
            subtitle_clip = TextClip(
                texto_subtitulo,
                fontsize=36,
                font="Arial-Bold",
                color='white',
                method='caption',
                align='center',
                size=(int(FINAL_WIDTH * 0.7), None)
            )

            subtitle_w, subtitle_h = subtitle_clip.size

            subtitle = (
                subtitle_clip
                .on_color(size=(FINAL_WIDTH, subtitle_h), color=(0, 0, 0), col_opacity=0.5)
                .set_duration(dur)
                .set_position(("center", FINAL_HEIGHT * 0.75))
            )

        zoom = lambda t: zoom_factor_start + (zoom_factor_end - zoom_factor_start) * (t / dur)

        animated_clip = (
            base_clip
            .resize(lambda t: zoom(t))
            .crop(
                width=FINAL_WIDTH,
                height=FINAL_HEIGHT,
                x_center=base_clip.w / 2,
                y_center=base_clip.h / 2
            )
        )

        composed = CompositeVideoClip([animated_clip, subtitle]) if SHOULD_INCLUDE_SUBTITLES else animated_clip
        clips.append(composed)
        print(f"  ✅ Clip {idx:03} listo para ensamblar")

    print(f"\n🔗 Concatenando {len(clips)} clips...")
    video = concatenate_videoclips(clips, method="compose")
    print(f"   Video concatenado: {video.duration:.2f}s")

    print(f"🎵 Cargando audio — narración: {narracion_path} | música: {musica_path}")
    audio_narracion = AudioFileClip(narracion_path)
    audio_musica = AudioFileClip(musica_path).volumex(0.2)
    print(f"   Narración: {audio_narracion.duration:.2f}s | Música: {audio_musica.duration:.2f}s")
    audio_musica = audio_musica.subclip(0, min(audio_narracion.duration, audio_musica.duration))

    audio_final = CompositeAudioClip([audio_musica, audio_narracion])
    video_final = video.set_audio(audio_final)

    print(f"💾 Exportando video → {output_path}")
    video_final.write_videofile(output_path, fps=24, codec='libx264', audio_codec='aac')


def generar_video_desde_story_id(story_id):
    story_dir = os.path.join(OUTPUT_DIR, story_id)
    image_dir = os.path.join(story_dir, "images")
    audio_dir = os.path.join(story_dir, "audios")
    narracion_path = os.path.join(audio_dir, "cuento_completo.mp3")
    json_path = os.path.join(story_dir, "story.json")
    final_video_path = os.path.join(story_dir, "video.mp4")

    if not all(os.path.exists(p) for p in [image_dir, audio_dir, narracion_path, json_path]):
        print("❌ Faltan archivos requeridos. Verificá que estén generadas las imágenes, audio, música y JSON.")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        historia_json = json.load(f)

    textos = historia_json["textos"]

    musica_path = download_music()

    if not musica_path or not os.path.exists(musica_path):
        print("❌ No se pudo obtener música válida. Abortando.")
        return

    # Calcular los tiempos de entrada y duración exactos de cada fragmento
    fragment_paths = [os.path.join(audio_dir, f"{idx:03}.mp3") for idx in range(1, len(textos) + 1)]

    # Leer duraciones individuales
    fragment_durations = [AudioFileClip(p).duration for p in fragment_paths]

    # Sumamos los silencios intermedios
    duraciones = []
    for i, d in enumerate(fragment_durations):
        if i != len(fragment_durations) - 1:
            d += SILENCIO_SEGUNDOS
        duraciones.append(d)


    generar_video(textos, duraciones, image_dir, narracion_path, musica_path, final_video_path)
    print(f"🎞️ Video generado desde datos existentes: {final_video_path}")

import sys

if __name__ == "__main__":
    start_time = time.time()
    parser = argparse.ArgumentParser(description="Generador de historias en video")

    parser.add_argument(
        "--ideas-file", type=str, default="general-history.json",
        help="Archivo JSON con las ideas base"
    )

    parser.add_argument("--story-id", type=str, help="ID de la historia generada (carpeta dentro de stories/)")
    parser.add_argument("--mode", type=str, default="video", help="Modo de ejecución: imagenes | audios | video | musica | juntar-audios | ralentizar")
    parser.add_argument("--dev", action="store_true", default=False, help="Modo dev: limita la historia a 4 secciones para probar el flujo rápido")

    args = parser.parse_args()
    IDEAS_FILE = "ideas/" + args.ideas_file
    story_id = args.story_id
    modo = args.mode
    DEV_MAX_SECTIONS = 4 if args.dev else None

    if args.dev:
        print(f"🛠️  Modo dev activo — máximo {DEV_MAX_SECTIONS} secciones")


    reintento = 0

    if story_id:
        print(f"📂 Usando historia ya generada: {story_id}")
        story_dir = os.path.join(OUTPUT_DIR, story_id)
        json_path = os.path.join(story_dir, "story.json")
        print(f"📄 Leyendo JSON desde: {json_path}")
        if not os.path.exists(json_path):
            print("❌ No se encontró story.json")
            exit(1)

        with open(json_path, "r", encoding="utf-8") as f:
            historia_json = json.load(f)
        if DEV_MAX_SECTIONS:
            historia_json["textos"]   = historia_json["textos"][:DEV_MAX_SECTIONS]
            historia_json["imagenes"] = historia_json["imagenes"][:DEV_MAX_SECTIONS]
        textos = historia_json["textos"]

        if modo == "imagenes":
            image_dir = os.path.join(story_dir, "images")
            audio_dir = os.path.join(story_dir, "audios")
            generar_imagenes(historia_json["imagenes"], image_dir, historia_json.get("contexto_visual_global"), audio_dir=audio_dir)

        elif modo == "audios":
            audio_dir = os.path.join(story_dir, "audios")
            generar_audios(historia_json["textos"], audio_dir)

        elif modo == "musica":
            musica_path = download_music()
            if musica_path:
                print(f"🎵 Música descargada: {musica_path}")
            else:
                print("❌ No se pudo descargar música válida.")

        elif modo == "video":
            generar_video_desde_story_id(story_id)

        elif modo == "ralentizar":
            image_dir = os.path.join(story_dir, "images")
            audio_dir = os.path.join(story_dir, "audios")
            duraciones = _calcular_duraciones_imagenes(historia_json["imagenes"], audio_dir=audio_dir)
            for idx, duracion in enumerate(duraciones, 1):
                raw_path = os.path.join(image_dir, f"{idx:03}_seedance_raw.mp4")
                out_path = os.path.join(image_dir, f"{idx:03}.mp4")
                if not os.path.exists(raw_path):
                    print(f"⏭️  Sin raw para imagen {idx:03}, saltando")
                    continue
                print(f"🐢 Ralentizando {idx:03}_seedance_raw.mp4 → {duracion:.1f}s")
                ralentizar_video(raw_path, duracion, out_path)
                print(f"✅ {idx:03}.mp4 listo")

        elif modo == "juntar-audios":
            audio_dir = os.path.join(story_dir, "audios")
            if not os.path.exists(audio_dir):
                print("❌ No existe la carpeta de audios")
                exit(1)

            num_fragmentos = len([f for f in os.listdir(audio_dir) if f.endswith(".mp3") and not f.startswith("cuento") and not f.startswith("silencio")])
            if num_fragmentos == 0:
                print("❌ No se encontraron fragmentos .mp3 para unir")
                exit(1)

            unir_audios_fragmentados(audio_dir, num_fragmentos)


        else:
            print("❌ Modo no reconocido. Usá uno de: imagenes | audios | video")
    else:
        while reintento < MAX_REINTENTOS:
            try:
                # Modo generación completa
                story_id = str(uuid.uuid4())[:8]
                story_dir = os.path.join(OUTPUT_DIR, story_id)
                image_dir = os.path.join(story_dir, "images")
                audio_dir = os.path.join(story_dir, "audios")
                os.makedirs(story_dir, exist_ok=True)

                idea = elegir_idea()
                print(f"🧠 Generando historia para: {idea['titulo']}")
                prompt = generar_prompt(idea)
                historia_json = extraer_json(llamar_llm(prompt))
                if DEV_MAX_SECTIONS:
                    historia_json["textos"]   = historia_json["textos"][:DEV_MAX_SECTIONS]
                    historia_json["imagenes"] = historia_json["imagenes"][:DEV_MAX_SECTIONS]
                guardar_historia_json(story_dir, idea, historia_json)

                narracion_path, duraciones = generar_audios(historia_json["textos"], audio_dir)
                duraciones_corregidas = []
                for i, d in enumerate(duraciones):
                    if i != len(duraciones) - 1:
                        d += SILENCIO_SEGUNDOS
                    duraciones_corregidas.append(d)

                generar_imagenes(historia_json["imagenes"], image_dir, historia_json.get("contexto_visual_global"), audio_dir=audio_dir)

                print(f"Duracion imágenes : {sum(duraciones_corregidas):.2f}s")
                print(f"Duracion audio solo: {AudioFileClip(narracion_path).duration:.2f}s")
                print(f"🕒 Duración total del video (con silencios): {sum(duraciones):.2f} segundos")
                musica_path = download_music()

                if not musica_path or not os.path.exists(musica_path):
                    print("❌ No se pudo descargar música válida. Abortando.")
                    exit(1)

                final_video_path = os.path.join(story_dir, "video.mp4")
                generar_video(historia_json["textos"], duraciones_corregidas, image_dir, narracion_path, musica_path, final_video_path)
                print(f"\n🎬 Video final generado: {final_video_path}")
                break

            except Exception as e:
                reintento += 1
                print(f"\n⚠️ Falló el intento {reintento} de {MAX_REINTENTOS}: {str(e)}")
                traceback.print_exc()
                if reintento >= MAX_REINTENTOS:
                    print("❌ Máximo de reintentos alcanzado. Abortando.")
                    exit(1)
                else:
                    print("🔄 Reintentando todo el proceso...\n")

    print(f"⏱️ Tiempo total de ejecución: {time.time() - start_time:.2f} segundos")

