import os
import json
from huggingface_hub import InferenceClient
from PIL import Image
from dotenv import load_dotenv

# === CONFIG ===
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN", "your_huggingface_token_here")  # Reemplaza con tu token de Hugging Face
INPUT_JSON = "stories/1.json"
OUTPUT_DIR = "imagenes_hf"

os.makedirs(OUTPUT_DIR, exist_ok=True)
client = InferenceClient(token=HF_TOKEN)

# === Leer JSON ===
with open(INPUT_JSON, encoding="utf-8") as f:
    data = json.load(f)
imagenes = data.get("imagenes", [])

# === Generar imágenes ===
for idx, img in enumerate(imagenes, 1):
    prompt = img["descripcion"]
    print(f"\n🖼️ Generando imagen {idx:03} → {prompt}")
    try:
        image = client.text_to_image(prompt)  # selecciona automáticamente un modelo válido
    except Exception as e:
        print("❌ Error:", e)
        continue

    output_path = os.path.join(OUTPUT_DIR, f"{idx:03}.png")
    image.save(output_path)
    print(f"✅ Imagen guardada en {output_path}")

print("\n✅ ¡Todas las imágenes generadas!")
