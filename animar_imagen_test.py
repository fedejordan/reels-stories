import os
from gradio_client import Client, handle_file
import shutil

# Parámetros
input_image_path = "test_input.png"
output_video_path = "test_output.mp4"
prompt = "make this image come alive, cinematic motion, smooth animation"
negative_prompt = (
    "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, "
    "static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, "
    "extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, "
    "fused fingers, still picture, messy background, watermark, text, signature"
)

client = Client("multimodalart/wan2-1-fast")

print("🚀 Enviando imagen a la API /generate_video...")
result = client.predict(
    input_image=handle_file(input_image_path),
    prompt=prompt,
    height=512,
    width=512,
    negative_prompt=negative_prompt,
    duration_seconds=2,
    guidance_scale=1,
    steps=4,
    seed=42,
    randomize_seed=True,
    api_name="/generate_video"
)

# Extraer y copiar video desde path local
video_path = result[0]["video"]
shutil.copy(video_path, output_video_path)
print(f"✅ Video copiado a {output_video_path}")
