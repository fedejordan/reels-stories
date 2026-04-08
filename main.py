import os
import re
import json
import uuid
import random
import subprocess
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from PIL import Image
from gtts import gTTS

if not hasattr(Image, "ANTIALIAS"):
    Image.ANTIALIAS = Image.Resampling.LANCZOS

from moviepy.editor import *
import yt_dlp
import requests
import time
from moviepy.video.tools.subtitles import SubtitlesClip
from moviepy.editor import TextClip
from elevenlabs.client import ElevenLabs
import traceback
from gradio_client import Client
from PIL import Image
from moviepy.editor import VideoFileClip
import argparse
from requests.exceptions import Timeout
from PIL import ImageDraw, ImageFont

# === CONFIGURACIÓN ===
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_ENDPOINT = "https://api.deepseek.com/v1/chat/completions"
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY")
SUNO_API_KEY    = os.getenv("SUNO_API_KEY")
GEMINI_API_KEY  = os.getenv("GEMINI_API_KEY")
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
MODO_ANIMADO = False  # Cambiar a False para usar imágenes estáticas
SHOULD_INCLUDE_SUBTITLES = True  # Cambiar a False si no se quieren subtítulos
SUBTITLE_AS_IMAGE = False

# ================================================================
# PROVIDER CONFIGURATION — cambiar estos valores para switching
# ================================================================

IMAGE_PROVIDER   = "hf-default"   # "flux" | "hf-default"
HF_IMAGE_MODEL   = "black-forest-labs/FLUX.1-schnell"
HF_IMAGE_WIDTH   = 768
HF_IMAGE_HEIGHT  = 1344           # 9:16 portrait

TTS_PROVIDER     = "elevenlabs"   # "elevenlabs" | "openai" | "gtts"
ELEVENLABS_VOICE_ID  = "80lPKtzJMPh1vjYMUgwe"
ELEVENLABS_MODEL_ID  = "eleven_multilingual_v2"
OPENAI_TTS_MODEL     = "tts-1-hd"
OPENAI_TTS_VOICE     = "onyx"

MUSIC_PROVIDER   = "local"        # "suno" | "local" | "ytdlp"
SUNO_POLL_INTERVAL = 5
SUNO_MAX_WAIT      = 300

LLM_PROVIDER     = "deepseek"     # "deepseek" | "openai" | "gemini"
DEEPSEEK_MODEL   = "deepseek-chat"
OPENAI_LLM_MODEL = "gpt-4o"
GEMINI_MODEL     = "gemini-2.0-flash"


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

3. `audio`: {"descripción del estilo/mood musical para generación automática, ej: 'instrumental épico cinematográfico, estilo Hans Zimmer'" if MUSIC_PROVIDER == "suno" else "nombre exacto de una pieza instrumental real (sin letra), disponible en YouTube, que intensifique el tono narrativo del video. Puede ser épico, melancólico, intrigante o inspirador según el caso."}

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

def _llm_openai(prompt):
    import openai
    oa_client = openai.OpenAI(api_key=OPENAI_API_KEY)
    response = oa_client.chat.completions.create(
        model=OPENAI_LLM_MODEL,
        messages=[
            {"role": "system", "content": "Sos un guionista experto en reels"},
            {"role": "user", "content": prompt},
        ],
        temperature=0.8,
    )
    return response.choices[0].message.content

def _llm_gemini(prompt):
    from google import genai as google_genai
    gc = google_genai.Client(api_key=GEMINI_API_KEY)
    response = gc.models.generate_content(model=GEMINI_MODEL, contents=prompt)
    return response.text

def llamar_llm(prompt):
    if LLM_PROVIDER == "deepseek":
        return _llm_deepseek(prompt)
    elif LLM_PROVIDER == "openai":
        return _llm_openai(prompt)
    elif LLM_PROVIDER == "gemini":
        return _llm_gemini(prompt)
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {LLM_PROVIDER}")

# Backwards compat alias
def llamar_a_deepseek(prompt):
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

def prepare_image_dimensions(input_path, base_height=512):
    with Image.open(input_path) as im:
        im = im.convert("RGB")
        orig_w, orig_h = im.size
        aspect_ratio = orig_w / orig_h

        new_height = base_height
        new_width = int(round(base_height * aspect_ratio))

        # Redondear al múltiplo de 32 más cercano
        new_width = (new_width // 32) * 32
        new_height = (new_height // 32) * 32

        resized = im.resize((new_width, new_height))
        temp_path = input_path.replace(".png", f"_{new_width}x{new_height}.png")
        resized.save(temp_path)

        return temp_path, new_width, new_height

def ralentizar_video(input_path, duracion_objetivo, output_path):
    clip = VideoFileClip(input_path)
    velocidad = clip.duration / duracion_objetivo
    if velocidad > 0:
        clip_ralentizado = clip.fx(vfx.speedx, factor=velocidad)
        clip_ralentizado = clip_ralentizado.set_duration(duracion_objetivo)
        clip_ralentizado.write_videofile(output_path, codec="libx264", audio=False, fps=24)
        clip_ralentizado.close()
    clip.close()
    
def animar_imagen(input_image_path, prompt, output_video_path, duracion):
    try:
        print(f"🎨 Animando imagen {input_image_path} con duración {duracion:.2f}s...")
        resized_img, width, height = prepare_image_dimensions(input_image_path)

        from gradio_client import handle_file
        import shutil

        result = client_wan.predict(
            input_image=handle_file(resized_img),
            prompt="make this image come alive, cinematic motion, smooth animation",
            height=height,
            width=width,
            negative_prompt=(
                "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, "
                "static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, "
                "extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, "
                "fused fingers, still picture, messy background, watermark, text, signature"
            ),
            duration_seconds=2,
            guidance_scale=1,
            steps=4,
            seed=42,
            randomize_seed=True,
            api_name="/generate_video"
        )

        video_path = result[0]["video"]
        temp_path = output_video_path.replace(".mp4", "_original.mp4")
        shutil.copy(video_path, temp_path)
        print(f"🎞️ Video base copiado: {temp_path}")

        ralentizar_video(temp_path, duracion, output_video_path)
        print(f"🐢 Video ralentizado guardado: {output_video_path}")

    except Exception as e:
        print("❌ Excepción al animar imagen:", e)



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

def generar_imagenes(imagenes, image_dir, contexto_visual_global=None, max_reintentos=10):
    os.makedirs(image_dir, exist_ok=True)
    hf_client = _build_hf_image_client()

    for idx, img in enumerate(imagenes, 1):
        prompt = f"Imagen a dibujar: {img['descripcion']}\n\nContexto: {contexto_visual_global}." if contexto_visual_global else img["descripcion"]
        intentos = 0
        exito = False

        while intentos < max_reintentos and not exito:
            print(f"🖼️ Generando imagen {idx:03} (intento {intentos + 1}) → {prompt}")
            try:
                image = _generate_single_image(hf_client, prompt)
                img_path = os.path.join(image_dir, f"{idx:03}.png")
                image.save(img_path)
                exito = True

                if MODO_ANIMADO:
                    anim_path = os.path.join(image_dir, f"{idx:03}.mp4")
                    duracion = img.get("milisegundos", 2000) / 1000.0
                    animar_imagen(img_path, prompt, anim_path, duracion)
            except Timeout:
                print(f"⚠️ Timeout alcanzado al generar imagen {idx:03}")
            except Exception as e:
                print(f"❌ Error generando imagen {idx:03}: {e}")
                intentos += 1
                if intentos < max_reintentos:
                    print("🔄 Reintentando...")
                    time.sleep(2)  # Espera entre intentos
                else:
                    print(f"🚫 No se pudo generar la imagen {idx:03} después de {max_reintentos} intentos.")





def _tts_elevenlabs(text, filename, elevenlabs_client):
    audio = elevenlabs_client.text_to_speech.convert(
        voice_id=ELEVENLABS_VOICE_ID,
        model_id=ELEVENLABS_MODEL_ID,
        text=text,
    )
    with open(filename, "wb") as f:
        for chunk in audio:
            f.write(chunk)

def _tts_openai(text, filename):
    import openai
    oa_client = openai.OpenAI(api_key=OPENAI_API_KEY)
    response = oa_client.audio.speech.create(
        model=OPENAI_TTS_MODEL,
        voice=OPENAI_TTS_VOICE,
        input=text,
    )
    response.stream_to_file(filename)

def _tts_gtts(text, filename):
    gTTS(text=text, lang="es").save(filename)

def _sintetizar_texto(text, filename, elevenlabs_client=None):
    if TTS_PROVIDER == "elevenlabs":
        _tts_elevenlabs(text, filename, elevenlabs_client)
    elif TTS_PROVIDER == "openai":
        _tts_openai(text, filename)
    elif TTS_PROVIDER == "gtts":
        _tts_gtts(text, filename)
    else:
        raise ValueError(f"Unknown TTS_PROVIDER: {TTS_PROVIDER}")

def generar_audios(textos, audio_dir):
    os.makedirs(audio_dir, exist_ok=True)

    elevenlabs_client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY")) if TTS_PROVIDER == "elevenlabs" else None

    # Crear un archivo de silencio de 0.5s si no existe
    silencio_path = os.path.join(audio_dir, "silencio.mp3")
    if not os.path.exists(silencio_path):
        subprocess.run([
            "ffmpeg", "-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono",
            "-t", str(SILENCIO_SEGUNDOS), "-q:a", "9", "-acodec", "libmp3lame", silencio_path
        ])

    audio_files = []
    durations = []

    for idx, fragmento in enumerate(textos, 1):
        texto = fragmento["texto"]
        filename = os.path.abspath(os.path.join(audio_dir, f"{idx:03}.mp3"))

        _sintetizar_texto(texto, filename, elevenlabs_client)

        audio_files.append(filename)

        audio_clip = AudioFileClip(filename)
        durations.append(audio_clip.duration)
        audio_clip.close()

        print(f"🎙️ Fragmento {idx:03} generado ({durations[-1]:.2f}s): {texto[:40]}...")

    # Concatenar todos los clips
    concat_path = os.path.join(audio_dir, "concat_list.txt")
    with open(concat_path, "w", encoding="utf-8") as f:
        for i, file in enumerate(audio_files):
            rel_path = os.path.relpath(file, audio_dir)
            f.write(f"file '{rel_path}'\n")
            if i != len(audio_files) - 1:  # no agregues silencio al final
                f.write(f"file 'silencio.mp3'\n")  # silencio está en el mismo dir


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

def _download_music_ytdlp(query, output_dir):
    if not query:
        print("No query para yt-dlp. Usando música local.")
        return _download_music_local()
    os.makedirs(output_dir, exist_ok=True)
    output_template = os.path.join(output_dir, "music")
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_template + ".%(ext)s",
        "quiet": False,
        "noplaylist": True,
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "192",
        }],
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.extract_info(f"ytsearch1:{query}", download=True)
    return output_template + ".mp3"

def _download_music_suno(query, output_dir):
    if not SUNO_API_KEY:
        print("SUNO_API_KEY no configurado. Usando música local.")
        return _download_music_local()

    os.makedirs(output_dir, exist_ok=True)
    headers = {
        "Authorization": f"Bearer {SUNO_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "prompt": query,
        "make_instrumental": True,
        "wait_audio": False,
    }
    try:
        resp = requests.post(
            "https://studio-api.suno.ai/api/generate/v2/",
            headers=headers, json=payload, timeout=30,
        )
        resp.raise_for_status()
        songs = resp.json()
        clip_id = songs[0]["id"]
    except Exception as e:
        print(f"Error al iniciar generación en Suno: {e}. Usando música local.")
        return _download_music_local()

    deadline = time.time() + SUNO_MAX_WAIT
    audio_url = None
    while time.time() < deadline:
        try:
            check = requests.get(
                f"https://studio-api.suno.ai/api/feed/?ids={clip_id}",
                headers=headers, timeout=15,
            )
            check.raise_for_status()
            item = check.json()[0]
            if item["status"] == "complete":
                audio_url = item["audio_url"]
                break
        except Exception as e:
            print(f"Error al consultar estado en Suno: {e}")
        time.sleep(SUNO_POLL_INTERVAL)

    if not audio_url:
        print("Suno generación excedió el tiempo límite. Usando música local.")
        return _download_music_local()

    output_path = os.path.join(output_dir, "music.mp3")
    try:
        audio_data = requests.get(audio_url, timeout=60)
        audio_data.raise_for_status()
        with open(output_path, "wb") as f:
            f.write(audio_data.content)
        return output_path
    except Exception as e:
        print(f"Error al descargar audio de Suno: {e}. Usando música local.")
        return _download_music_local()

def download_music(query=None, output_dir=None):
    if MUSIC_PROVIDER == "suno":
        return _download_music_suno(query, output_dir)
    elif MUSIC_PROVIDER == "ytdlp":
        return _download_music_ytdlp(query, output_dir)
    else:
        return _download_music_local()

def crear_subtitulo_como_imagen(texto, width, fontsize=36):
    # Crear imagen base
    img = Image.new("RGBA", (width, 500), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Intentar cargar una fuente compatible
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", fontsize)
    except:
        font = ImageFont.load_default()

    # Calcular tamaño de texto
    lines = []
    words = texto.split()
    line = ""

    for word in words:
        test_line = f"{line} {word}".strip()
        if draw.textlength(test_line, font=font) <= width * 0.9:
            line = test_line
        else:
            lines.append(line)
            line = word
    lines.append(line)

    total_height = len(lines) * (fontsize + 10)
    img = Image.new("RGBA", (width, total_height), (0, 0, 0, 128))  # fondo semitransparente
    draw = ImageDraw.Draw(img)

    for i, line in enumerate(lines):
        w = draw.textlength(line, font=font)
        draw.text(((width - w) / 2, i * (fontsize + 10)), line, font=font, fill=(255, 255, 255, 255))

    temp_path = f"/tmp/subtitle_{uuid.uuid4().hex}.png"
    img.save(temp_path)
    return temp_path


def generar_video(textos, duraciones, image_dir, narracion_path, musica_path, output_path):
    clips = []

    for idx, dur in enumerate(duraciones, 1):
        if MODO_ANIMADO:
            video_path = os.path.join(image_dir, f"{idx:03}.mp4")
            base_clip = VideoFileClip(video_path).subclip(0, dur).resize(height=1920)
        else:
            img_path = os.path.join(image_dir, f"{idx:03}.png")
            base_clip = ImageClip(img_path, duration=dur).resize(height=1920).fadein(0.2).fadeout(0.2)


        # Zoom aleatorio: in (acercar) o out (alejar)
        zoom_type = "in" #random.choice(["in", "out"])
        zoom_factor_start = 1.0 if zoom_type == "in" else 1.1
        zoom_factor_end = 1.1 if zoom_type == "in" else 1.0

        # Texto del fragmento correspondiente
        texto_subtitulo = textos[idx - 1]["texto"]

        if SUBTITLE_AS_IMAGE:
            subtitle_path = crear_subtitulo_como_imagen(texto_subtitulo, FINAL_WIDTH)
            subtitle = (
                ImageClip(subtitle_path)
                .set_duration(dur)
                .set_position(("center", FINAL_HEIGHT * 0.75))  # 70% desde arriba
            )
        else:      
            # Crear clip con subtítulo
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
                .set_position(("center", FINAL_HEIGHT * 0.75))  # 70% desde arriba
            )

        # Zoom centrado (resize + crop para evitar bordes negros)
        zoom = lambda t: zoom_factor_start + (zoom_factor_end - zoom_factor_start) * (t / dur)

        # Escalamos y centramos para mantener movimiento y relación 9:16
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

    video = concatenate_videoclips(clips, method="compose")

    audio_narracion = AudioFileClip(narracion_path)
    audio_musica = AudioFileClip(musica_path).volumex(0.2)
    audio_musica = audio_musica.subclip(0, min(audio_narracion.duration, audio_musica.duration))

    # Combinar audio
    audio_final = CompositeAudioClip([audio_musica, audio_narracion])
    video_final = video.set_audio(audio_final)

    # Exportar video
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

    musica_path = download_music(historia_json.get("audio"), os.path.join(story_dir, "music"))

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

    parser.add_argument(
        "--subtitle-as-image", action="store_true",
        default=False,
        help="Generar subtítulos como imágenes en lugar de texto superpuesto"
    )

    parser.add_argument("--story-id", type=str, help="ID de la historia generada (carpeta dentro de stories/)")
    parser.add_argument("--mode", type=str, default="video", help="Modo de ejecución: imagenes | audios | video | musica | juntar-audios")

    args = parser.parse_args()
    IDEAS_FILE = "ideas/" + args.ideas_file
    SUBTITLE_AS_IMAGE = args.subtitle_as_image
    story_id = args.story_id
    modo = args.mode


    reintento = 0

    if MODO_ANIMADO:
        client_wan = Client("multimodalart/wan2-1-fast")  # carga remota correcta

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
        textos = historia_json["textos"]

        if modo == "imagenes":
            image_dir = os.path.join(story_dir, "images")
            generar_imagenes(historia_json["imagenes"], image_dir, historia_json.get("contexto_visual_global"))

        elif modo == "audios":
            audio_dir = os.path.join(story_dir, "audios")
            generar_audios(historia_json["textos"], audio_dir)

        elif modo == "musica":
            musica_path = download_music(historia_json.get("audio"), os.path.join(story_dir, "music"))
            if musica_path:
                print(f"🎵 Música descargada: {musica_path}")
            else:
                print("❌ No se pudo descargar música válida.")

        elif modo == "video":
            generar_video_desde_story_id(story_id)

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
                guardar_historia_json(story_dir, idea, historia_json)

                generar_imagenes(historia_json["imagenes"], image_dir, historia_json.get("contexto_visual_global"))
                narracion_path, duraciones = generar_audios(historia_json["textos"], audio_dir)
                duraciones_corregidas = []
                for i, d in enumerate(duraciones):
                    if i != len(duraciones) - 1:          # silencio entre fragmentos
                        d += SILENCIO_SEGUNDOS
                    duraciones_corregidas.append(d)

                print(f"Duracion imágenes : {sum(duraciones_corregidas):.2f}s")
                print(f"Duracion audio solo: {AudioFileClip(narracion_path).duration:.2f}s")
                print(f"🕒 Duración total del video (con silencios): {sum(duraciones):.2f} segundos")
                musica_path = download_music(historia_json["audio"], os.path.join(story_dir, "music"))

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

