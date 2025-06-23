import json
import requests
import os
import random
import uuid
from dotenv import load_dotenv

load_dotenv()
# CONFIG
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_ENDPOINT = "https://api.deepseek.com/v1/chat/completions"
IDEAS_FILE = "ideas.json"
OUTPUT_DIR = "stories"

PROMPT_TEMPLATE = """
Quiero que generes un video corto estilo "reel" de 30 segundos, basado en la siguiente idea de historia fantasiosa:

{descripcion}

El video debe estar compuesto por tres elementos:

1. `textos`: una lista ordenada de fragmentos narrativos que serán leídos en voz en off. Cada texto debe tener su duración en milisegundos (`milisegundos`) y el contenido del texto (`texto`). El total debe sumar aproximadamente 30,000 milisegundos (30 segundos).

2. `imagenes`: una lista de descripciones de imágenes que se mostrarán junto al texto. Cada una debe tener una duración (`milisegundos`) y una descripción visual creativa (`descripcion`) que represente lo que se narra en ese momento.

3. `audio`: el nombre de una pieza musical instrumental conocida (sin letra) que acompañe el tono de la historia. Puede ser clásica, épica, ambiental o melancólica, según lo que mejor se ajuste.

El formato de salida debe ser estrictamente este JSON:

```json
{{
  "textos": [
    {{
      "milisegundos": 4000,
      "texto": "Ejemplo de texto narrado..."
    }},
    ...
  ],
  "imagenes": [
    {{
      "milisegundos": 4000,
      "descripcion": "Ejemplo de imagen..."
    }},
    ...
  ],
  "audio": "Nombre de la pieza musical instrumental"
}}
"""

def elegir_idea():
    with open(IDEAS_FILE, "r", encoding="utf-8") as f:
        ideas = json.load(f)
    return random.choice(ideas)

def generar_prompt(idea):
    return PROMPT_TEMPLATE.format(descripcion=idea["descripcion"])

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
    content = response.json()
    return content["choices"][0]["message"]["content"]

def extraer_json(respuesta):
    inicio = respuesta.find("{")
    fin = respuesta.rfind("}") + 1
    json_str = respuesta[inicio:fin]
    return json.loads(json_str)

def guardar_historia(idea, contenido_json):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    story_id = str(uuid.uuid4())[:8]
    filepath = os.path.join(OUTPUT_DIR, f"{story_id}.json")
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump({
            "titulo": idea["titulo"],
            "descripcion": idea["descripcion"],
            "video": contenido_json
            }, f, ensure_ascii=False, indent=2)
    print(f"✅ Historia guardada en: {filepath}")

if __name__ == "__main__":
    idea = elegir_idea()
    prompt = generar_prompt(idea)
    print(f"🎬 Generando historia para: {idea['titulo']}")
    respuesta = llamar_a_deepseek(prompt)
    video_json = extraer_json(respuesta)
    guardar_historia(idea, video_json)