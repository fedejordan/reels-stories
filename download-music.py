import yt_dlp
import json
import os
import re

def sanitize_filename(text):
    # Convierte texto en un nombre de archivo válido
    return re.sub(r'[^a-zA-Z0-9_-]', '_', text).lower()

def download_audio(query, output_path="audio.mp3"):
    search_url = f"ytsearch1:{query}"
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': output_path + ".%(ext)s",  # Se guarda como MP3
        'quiet': False,
        'noplaylist': True,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(search_url, download=True)
        print(f"\n✅ Audio descargado como: {output_path}.mp3")

if __name__ == "__main__":
    # Ruta al archivo JSON
    json_path = "stories/1.json"

    # Cargar el JSON
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Obtener el título de audio
    audio_title = data.get("audio", "").strip()

    if not audio_title:
        print("❌ No se encontró el campo 'audio' en el JSON.")
    else:
        output_filename = sanitize_filename(audio_title)
        output_filename = "music"
        download_audio(audio_title, f"audios/{output_filename}")
