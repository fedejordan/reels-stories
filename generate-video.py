import os
import json
import subprocess
from moviepy.editor import *

# === CONFIGURACIÓN ===
STORY_JSON = "stories/1.json"
AUDIO_NARRACION = "audios/cuento_completo.mp3"
AUDIO_MUSICA = "audios/music.mp3"
IMG_DIR = "imagenes_hf"
OUTPUT_VIDEO = "video_final.mp4"

# === CARGAR DATOS ===
with open(STORY_JSON, "r", encoding="utf-8") as f:
    data = json.load(f)

imagenes = data["imagenes"]
durations = [img["milisegundos"] / 1000 for img in imagenes]

# === CREAR CLIPS DE IMAGEN ===
image_clips = []
for idx, dur in enumerate(durations, 1):
    path = os.path.join(IMG_DIR, f"{idx:03}.png")
    clip = ImageClip(path).set_duration(dur).fadein(0.2).fadeout(0.2)
    image_clips.append(clip)

video = concatenate_videoclips(image_clips, method="compose")

# === AGREGAR AUDIO NARRADO ===
audio_narracion = AudioFileClip(AUDIO_NARRACION)

# === OPCIONAL: MÚSICA DE FONDO MEZCLADA SUAVEMENTE ===
if AUDIO_MUSICA:
    audio_musica = AudioFileClip(AUDIO_MUSICA).volumex(0.2)
    # Cortar música al largo exacto de la narración (evita el error)
    audio_musica = audio_musica.subclip(0, min(audio_musica.duration, audio_narracion.duration))
    audio_final = CompositeAudioClip([audio_musica, audio_narracion])
else:
    audio_final = audio_narracion


video = video.set_audio(audio_final)

# === EXPORTAR VIDEO FINAL ===
video.write_videofile(OUTPUT_VIDEO, fps=24, codec='libx264', audio_codec='aac')

print(f"\n🎬 Video generado: {OUTPUT_VIDEO}")
