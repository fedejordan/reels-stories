import os
import json
import sys
from dotenv import load_dotenv

load_dotenv()
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_ENDPOINT = "https://api.deepseek.com/v1/chat/completions"

def generar_prompt(story):
    resumen = story["descripcion"]
    return f"""
Actuá como experto en redes sociales virales, especialmente TikTok. Tu tarea es crear una descripción atractiva para un video corto (reel) basado en la siguiente historia real:

Título: {story['titulo']}
Resumen: {resumen}

Generá un texto corto (menos de 300 caracteres) para publicar junto al video, que incluya:
- Una frase que atrape (curiosa, impactante o emotiva)
- Hashtags relevantes (#historia #tiktokcultural #mariecurie etc.)
- Opcional: un título breve para destacar el video

La descripción debe invitar a ver el video completo y conectar emocionalmente con el espectador.
"""

def llamar_a_deepseek(prompt):
    import requests

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
    }

    data = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
    }

    response = requests.post(DEEPSEEK_ENDPOINT, headers=headers, json=data)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]

def main():
    if len(sys.argv) < 2:
        print("Uso: python generar_descripcion_tiktok.py <story_id>")
        sys.exit(1)

    story_id = sys.argv[1]
    path_json = f"stories/{story_id}/story.json"
    path_output = f"stories/{story_id}/tiktok.txt"

    if not os.path.exists(path_json):
        print(f"No se encontró el archivo: {path_json}")
        sys.exit(1)

    with open(path_json, "r", encoding="utf-8") as f:
        story = json.load(f)

    prompt = generar_prompt(story)
    try:
        descripcion = llamar_a_deepseek(prompt)
    except Exception as e:
        print("❌ Error al llamar a DeepSeek:", e)
        sys.exit(1)

    with open(path_output, "w", encoding="utf-8") as f:
        f.write(descripcion.strip())

    print(f"✅ Descripción generada en {path_output}")

if __name__ == "__main__":
    main()
