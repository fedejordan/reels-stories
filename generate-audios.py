from gtts import gTTS
import os
import json
import subprocess

# Leer el cuento desde JSON
with open("stories/1.json", "r", encoding="utf-8") as f:
    cuento = json.load(f)

# Crear carpeta de salida
output_dir = "audios"
os.makedirs(output_dir, exist_ok=True)

# Generar audios
audio_files = []
for idx, fragmento in enumerate(cuento["textos"], 1):
    texto = fragmento["texto"]
    filename = os.path.abspath(os.path.join(output_dir, f"{idx:03}.mp3"))
    tts = gTTS(text=texto, lang='es')
    tts.save(filename)
    audio_files.append(filename)
    print(f"Generado: {filename}")

# Crear archivo de texto con lista de audios (rutas absolutas)
concat_list_path = os.path.join(output_dir, "concat_list.txt")
with open(concat_list_path, "w", encoding="utf-8") as f:
    for filepath in audio_files:
        f.write(f"file '{filepath}'\n")

# Unir los archivos con ffmpeg
final_audio_path = os.path.join(output_dir, "cuento_completo.mp3")
subprocess.run([
    "ffmpeg", "-f", "concat", "-safe", "0",
    "-i", concat_list_path,
    "-c", "copy", final_audio_path
])

print(f"\n✅ Archivo final generado: {final_audio_path}")

# Reproducir (solo Mac)
os.system(f"afplay '{final_audio_path}'")
