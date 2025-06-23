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
from moviepy.editor import *
import yt_dlp
import requests
import time

# === CONFIGURACIÓN ===
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_ENDPOINT = "https://api.deepseek.com/v1/chat/completions"
IDEAS_FILE = "ideas.json"
OUTPUT_DIR = "stories"

def sanitize_filename(text):
    return re.sub(r'[^a-zA-Z0-9_-]', '_', text).lower()

def elegir_idea():
    with open(IDEAS_FILE, "r", encoding="utf-8") as f:
        ideas = json.load(f)
    return random.choice(ideas)

def generar_prompt(idea):
    return f"""
Quiero que generes un video corto estilo "reel" de 30 segundos, basado en la siguiente idea de historia fantasiosa:

{idea['descripcion']}

El video debe estar compuesto por tres elementos:

1. `textos`: una lista ordenada de fragmentos narrativos que serán leídos en voz en off. Cada texto debe tener su duración en milisegundos (`milisegundos`) y el contenido del texto (`texto`). La narrativa debe desarrollarse progresivamente, manteniendo una estructura clara de introducción, desarrollo y cierre.

2. `imagenes`: una lista de descripciones visuales para acompañar cada fragmento de texto. Cada una debe tener su campo `milisegundos` (inicio) y `descripcion`. Es fundamental que TODAS las imágenes mantengan un estilo visual coherente entre sí (por ejemplo: realista, animación digital, dibujo a mano, estilo oscuro, mágico, etc.). Además, deben reflejar el mismo tono emocional y la atmósfera general de la historia (por ejemplo: tenebroso, épico, melancólico, fantástico).

    Las descripciones de imagen deben estar contextualizadas dentro del universo de la historia, reflejando personajes, lugares o transformaciones que ya se mencionaron o se intuyen.

3. `audio`: el nombre exacto de una pieza musical instrumental conocida y disponible en YouTube (sin letra). Debe ser una canción real y buscable como: "Time - Hans Zimmer", "Clair de Lune - Debussy", "Lux Aeterna - Clint Mansell". No incluyas descripciones ni efectos de sonido.

Formato de salida (JSON):
{{"textos":[{{"milisegundos":0,"texto":"..."}}], "imagenes":[{{"milisegundos":0,"descripcion":"..."}}], "audio":"..."}}"""

def llamar_a_deepseek(prompt):
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": "Sos un guionista experto en reels"},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.8
    }
    response = requests.post(DEEPSEEK_ENDPOINT, headers=headers, json=payload)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]

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

def generar_imagenes(imagenes, image_dir):
    os.makedirs(image_dir, exist_ok=True)
    client = InferenceClient(token=HF_TOKEN)
    for idx, img in enumerate(imagenes, 1):
        prompt = img["descripcion"]
        print(f"🖼️ Generando imagen {idx:03} → {prompt}")
        try:
            image = client.text_to_image(prompt)
            image.save(os.path.join(image_dir, f"{idx:03}.png"))
        except Exception as e:
            print("❌ Error generando imagen:", e)

def generar_audios(textos, audio_dir):
    os.makedirs(audio_dir, exist_ok=True)
    audio_files = []
    durations = []

    for idx, fragmento in enumerate(textos, 1):
        texto = fragmento["texto"]
        filename = os.path.abspath(os.path.join(audio_dir, f"{idx:03}.mp3"))
        gTTS(text=texto, lang='es').save(filename)
        audio_files.append(filename)

        audio_clip = AudioFileClip(filename)
        durations.append(audio_clip.duration)
        audio_clip.close()

        print(f"🎙️ Fragmento {idx:03} generado ({durations[-1]:.2f}s): {texto[:40]}...")

    concat_path = os.path.join(audio_dir, "concat_list.txt")
    with open(concat_path, "w", encoding="utf-8") as f:
        for file in audio_files:
            f.write(f"file '{file}'\n")

    final_audio = os.path.join(audio_dir, "cuento_completo.mp3")
    subprocess.run(["ffmpeg", "-f", "concat", "-safe", "0", "-i", concat_path, "-c", "copy", final_audio])
    return final_audio, durations

def download_music(query, output_path, max_duration_sec=600):
    print(f"🔍 Buscando música: {query}")
    ydl_opts_info = {
        'format': 'bestaudio/best',
        'quiet': True,
        'noplaylist': True,
    }

    with yt_dlp.YoutubeDL(ydl_opts_info) as ydl:
        results = ydl.extract_info(f"ytsearch10:{query}", download=False)["entries"]

    for entry in results:
        if entry.get("duration", 0) <= max_duration_sec:
            print(f"🎵 Descargando: {entry['title']} ({entry['duration']} segundos)")
            ydl_opts_download = {
                'format': 'bestaudio/best',
                'outtmpl': output_path + ".%(ext)s",
                'quiet': True,
                'noplaylist': True,
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'mp3',
                    'preferredquality': '192',
                }],
            }
            with yt_dlp.YoutubeDL(ydl_opts_download) as ydl:
                ydl.download([entry["webpage_url"]])
            return output_path + ".mp3"

    print("❌ No se encontró música válida.")
    return None

from moviepy.video.fx.all import resize

def generar_video(imagenes, duraciones, image_dir, narracion_path, musica_path, output_path):
    clips = []

    for idx, dur in enumerate(duraciones, 1):
        img_path = os.path.join(image_dir, f"{idx:03}.png")
        base_clip = ImageClip(img_path, duration=dur).fadein(0.2).fadeout(0.2)

        # Zoom aleatorio: in (acercar) o out (alejar)
        zoom_type = random.choice(["in", "out"])
        zoom_factor_start = 1.0 if zoom_type == "in" else 1.1
        zoom_factor_end = 1.1 if zoom_type == "in" else 1.0

        animated_clip = base_clip.resize(lambda t: zoom_factor_start + (zoom_factor_end - zoom_factor_start) * (t / dur))
        clips.append(animated_clip)

    video = concatenate_videoclips(clips, method="compose")

    audio_narracion = AudioFileClip(narracion_path)
    audio_musica = AudioFileClip(musica_path).volumex(0.2)
    audio_musica = audio_musica.subclip(0, min(audio_narracion.duration, audio_musica.duration))

    video = video.set_audio(CompositeAudioClip([audio_musica, audio_narracion]))
    video.write_videofile(output_path, fps=24, codec='libx264', audio_codec='aac')


# === EJECUCIÓN PRINCIPAL ===
if __name__ == "__main__":
    start_time = time.time()

    # Preparar estructura de carpetas por historia
    story_id = str(uuid.uuid4())[:8]
    story_dir = os.path.join(OUTPUT_DIR, story_id)
    image_dir = os.path.join(story_dir, "images")
    audio_dir = os.path.join(story_dir, "audios")
    os.makedirs(story_dir, exist_ok=True)

    # Generar historia
    idea = elegir_idea()
    print(f"🧠 Generando historia para: {idea['titulo']}")
    prompt = generar_prompt(idea)
    historia_json = extraer_json(llamar_a_deepseek(prompt))
    guardar_historia_json(story_dir, idea, historia_json)

    # Generar imágenes, audios y música
    generar_imagenes(historia_json["imagenes"], image_dir)
    narracion_path, duraciones = generar_audios(historia_json["textos"], audio_dir)
    musica_path = download_music(historia_json["audio"], os.path.join(story_dir, "music"))

    if not musica_path or not os.path.exists(musica_path):
        print("❌ No se pudo descargar música válida. Abortando.")
        exit(1)

    # Generar video final
    final_video_path = os.path.join(story_dir, "video.mp4")
    generar_video(historia_json["imagenes"], duraciones, image_dir, narracion_path, musica_path, final_video_path)

    print(f"\n🎬 Video final generado: {final_video_path}")
    print(f"⏱️ Tiempo total de ejecución: {time.time() - start_time:.2f} segundos")
