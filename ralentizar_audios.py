import os
import subprocess
import sys

OUTPUT_DIR = "stories"
ATEMPO = 0.9  # velocidad: <1 más lento, >1 más rápido

def ralentizar_audios(story_id, atempo=ATEMPO):
    story_dir = os.path.join(OUTPUT_DIR, story_id)
    audio_dir = os.path.join(story_dir, "audios")
    silencio_path = os.path.join(audio_dir, "silencio.mp3")
    cuento_completo_path = os.path.join(audio_dir, "cuento_completo.mp3")

    if not os.path.isdir(audio_dir):
        print(f"❌ No existe la carpeta de audios: {audio_dir}")
        return

    files = sorted(f for f in os.listdir(audio_dir) if f.endswith(".mp3") and f not in ["cuento_completo.mp3", "silencio.mp3"])
    if not files:
        print("⚠️ No se encontraron archivos .mp3 para ralentizar.")
        return

    print(f"🐢 Ralentizando {len(files)} audios...")
    for filename in files:
        input_path = os.path.join(audio_dir, filename)
        temp_path = os.path.join(audio_dir, "temp_" + filename)

        subprocess.run([
            "ffmpeg", "-y", "-i", input_path,
            "-filter:a", f"atempo={atempo}",
            temp_path
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        os.replace(temp_path, input_path)
        print(f"✅ Ralentizado: {filename}")

    # Regenerar cuento_completo.mp3
    print("🔄 Regenerando cuento_completo.mp3...")
    concat_list_path = os.path.join(audio_dir, "concat_list.txt")
    with open(concat_list_path, "w", encoding="utf-8") as f:
        for i, file in enumerate(files):
            f.write(f"file '{os.path.join(audio_dir, file)}'\n")
            if i != len(files) - 1 and os.path.exists(silencio_path):
                f.write(f"file '{silencio_path}'\n")

    subprocess.run([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", concat_list_path,
        "-c", "copy", cuento_completo_path
    ])

    print(f"🎧 Narración final regenerada: {cuento_completo_path}")
    print("✅ Proceso completado.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Uso: python ralentizar_audios.py <story_id>")
    else:
        ralentizar_audios(sys.argv[1])
