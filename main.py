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
from moviepy.video.tools.subtitles import SubtitlesClip
from moviepy.editor import TextClip
from moviepy.video.fx.all import resize
from elevenlabs.client import ElevenLabs

# === CONFIGURACIÓN ===
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_ENDPOINT = "https://api.deepseek.com/v1/chat/completions"
IDEAS_FILE = "ideas.json"
OUTPUT_DIR = "stories"
os.environ["IMAGEMAGICK_BINARY"] = "/opt/homebrew/bin/convert"  # o el path que te dé `which convert`
FINAL_WIDTH = 1080
FINAL_HEIGHT = 1920
client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))


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

⚠️ Además, generá un campo extra llamado `contexto_visual_global`, donde describas de forma clara y unificada:
- el **estilo gráfico común** (realismo, animación digital, acuarela, etc)
- características de los **personajes principales** (edad, género, expresión, vestimenta)
- ambientación general (época, lugar, iluminación, clima)
Este campo se usará para mantener la coherencia visual en todas las imágenes.

Formato de salida:
{{"textos":[{{"milisegundos":0,"texto":"..."}}], "imagenes":[{{"milisegundos":0,"descripcion":"..."}}], "audio":"...", "contexto_visual_global": "..."}}"""

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

def generar_imagenes(imagenes, image_dir, contexto_visual_global=None):
    os.makedirs(image_dir, exist_ok=True)
    client = InferenceClient(token=HF_TOKEN)
    for idx, img in enumerate(imagenes, 1):
        prompt = f"{contexto_visual_global}. {img['descripcion']}" if contexto_visual_global else img["descripcion"]
        print(f"🖼️ Generando imagen {idx:03} → {prompt}")
        try:
            image = client.text_to_image(prompt)
            image.save(os.path.join(image_dir, f"{idx:03}.png"))
        except Exception as e:
            print("❌ Error generando imagen:", e)



def generar_audios(textos, audio_dir):
    os.makedirs(audio_dir, exist_ok=True)

    # Crear un archivo de silencio de 0.5s si no existe
    silencio_path = os.path.join(audio_dir, "silencio.mp3")
    if not os.path.exists(silencio_path):
        subprocess.run([
            "ffmpeg", "-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono",
            "-t", "0.5", "-q:a", "9", "-acodec", "libmp3lame", silencio_path
        ])


    audio_files = []
    durations = []

    VOICE_ID = "80lPKtzJMPh1vjYMUgwe"  # o "Bella", "Adam", u otra voz preexistente

    for idx, fragmento in enumerate(textos, 1):
        texto = fragmento["texto"]
        filename = os.path.abspath(os.path.join(audio_dir, f"{idx:03}.mp3"))

        audio = client.text_to_speech.convert(
            voice_id=VOICE_ID,
            model_id="eleven_multilingual_v2",
            text=texto
        )

        with open(filename, "wb") as f:
            for chunk in audio:
                f.write(chunk)
        
        # Ralentizar el audio usando FFmpeg (tempo < 1.0 para hacerlo más lento)
        slow_filename = filename.replace(".mp3", "_slow.mp3")
        subprocess.run([
            "ffmpeg", "-i", filename,
            "-filter:a", "atempo=0.9",
            "-y", slow_filename
        ])

        # Usar la versión ralentizada
        os.replace(slow_filename, filename)

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
            "-t", "0.5", "-q:a", "9", "-acodec", "libmp3lame", "silencio.mp3"
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

def generar_video(textos, duraciones, image_dir, narracion_path, musica_path, output_path):
    clips = []

    for idx, dur in enumerate(duraciones, 1):
        img_path = os.path.join(image_dir, f"{idx:03}.png")
        base_clip = ImageClip(img_path, duration=dur).resize(height=1920).fadein(0.2).fadeout(0.2)

        # Zoom aleatorio: in (acercar) o out (alejar)
        zoom_type = random.choice(["in", "out"])
        zoom_factor_start = 1.0 if zoom_type == "in" else 1.1
        zoom_factor_end = 1.1 if zoom_type == "in" else 1.0

        # Texto del fragmento correspondiente
        texto_subtitulo = textos[idx - 1]["texto"]

        # Crear clip con subtítulo
        subtitle_clip = TextClip(
            texto_subtitulo,
            fontsize=32,
            font="Arial-Bold",
            color='white',
            method='caption',
            size=(int(FINAL_WIDTH * 0.9), None),
            align='center'
        )

        subtitle_w, subtitle_h = subtitle_clip.size

        subtitle = (
            subtitle_clip
            .on_color(size=(FINAL_WIDTH, subtitle_h), color=(0, 0, 0), col_opacity=0.5)
            .set_duration(dur)
            .set_position(("center", FINAL_HEIGHT - subtitle_h - 100))  # al fondo sin salir
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

        composed = CompositeVideoClip([animated_clip, subtitle])

        clips.append(composed)

    video = concatenate_videoclips(clips, method="compose")

    audio_narracion = AudioFileClip(narracion_path)
    audio_musica = AudioFileClip(musica_path).volumex(0.2)
    audio_musica = audio_musica.subclip(0, min(audio_narracion.duration, audio_musica.duration))

    video = video.set_audio(CompositeAudioClip([audio_musica, audio_narracion]))
    video.write_videofile(output_path, fps=24, codec='libx264', audio_codec='aac')

def generar_video_desde_story_id(story_id):
    story_dir = os.path.join(OUTPUT_DIR, story_id)
    image_dir = os.path.join(story_dir, "images")
    audio_dir = os.path.join(story_dir, "audios")
    musica_path = os.path.join(story_dir, "music.mp3")
    narracion_path = os.path.join(audio_dir, "cuento_completo.mp3")
    json_path = os.path.join(story_dir, "story.json")
    final_video_path = os.path.join(story_dir, "video.mp4")

    if not all(os.path.exists(p) for p in [image_dir, audio_dir, musica_path, narracion_path, json_path]):
        print("❌ Faltan archivos requeridos. Verificá que estén generadas las imágenes, audio, música y JSON.")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        historia_json = json.load(f)

    textos = historia_json["textos"]

    # Calcular duraciones de los fragmentos ya grabados
    duraciones = []
    for idx in range(1, len(textos) + 1):
        audio_clip = AudioFileClip(os.path.join(audio_dir, f"{idx:03}.mp3"))
        duraciones.append(audio_clip.duration)
        audio_clip.close()

    generar_video(textos, duraciones, image_dir, narracion_path, musica_path, final_video_path)
    print(f"🎞️ Video generado desde datos existentes: {final_video_path}")


import sys

if __name__ == "__main__":
    start_time = time.time()

    if len(sys.argv) >= 2:
        story_id = sys.argv[1]
        modo = sys.argv[2] if len(sys.argv) > 2 else "video"  # default = generar video
        print(f"📂 Usando historia ya generada: {story_id}")
        story_dir = os.path.join(OUTPUT_DIR, story_id)
        json_path = os.path.join(story_dir, "story.json")

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
            musica_path = download_music(historia_json["audio"], os.path.join(story_dir, "music"))
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
        # Modo generación completa
        story_id = str(uuid.uuid4())[:8]
        story_dir = os.path.join(OUTPUT_DIR, story_id)
        image_dir = os.path.join(story_dir, "images")
        audio_dir = os.path.join(story_dir, "audios")
        os.makedirs(story_dir, exist_ok=True)

        idea = elegir_idea()
        print(f"🧠 Generando historia para: {idea['titulo']}")
        prompt = generar_prompt(idea)
        historia_json = extraer_json(llamar_a_deepseek(prompt))
        guardar_historia_json(story_dir, idea, historia_json)

        generar_imagenes(historia_json["imagenes"], image_dir, historia_json.get("contexto_visual_global"))
        narracion_path, duraciones = generar_audios(historia_json["textos"], audio_dir)
        musica_path = download_music(historia_json["audio"], os.path.join(story_dir, "music"))

        if not musica_path or not os.path.exists(musica_path):
            print("❌ No se pudo descargar música válida. Abortando.")
            exit(1)

        final_video_path = os.path.join(story_dir, "video.mp4")
        generar_video(historia_json["textos"], duraciones, image_dir, narracion_path, musica_path, final_video_path)
        print(f"\n🎬 Video final generado: {final_video_path}")

    print(f"⏱️ Tiempo total de ejecución: {time.time() - start_time:.2f} segundos")

