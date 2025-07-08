# 🎮 ReelGenerator: Historias en Video Automatizadas

Este proyecto genera **videos estilo "reel"** (verticales, de 2 minutos) basados en historias reales o creativas, combinando **narrativa, imágenes generadas por IA, audio, música y subtítulos**.

---

## ✨ ¿Qué hace este proyecto?

A partir de un archivo JSON con historias (`ideas/*.json`), el sistema:

1. Genera un guión narrativo estructurado (texto + imágenes + música) con IA (DeepSeek).
2. Crea imágenes realistas alineadas con la narrativa usando modelos de Hugging Face.
3. Narra los textos con ElevenLabs o gTTS.
4. Añade música instrumental apropiada.
5. Produce un video final con subtítulos, movimiento, zoom y transiciones cinematográficas.

---

## 🛠️ Requisitos

### Python

* Python 3.9 o superior

### Librerías

Instalación automática con:

```bash
pip install -r requirements.txt
```

Incluye:

* `moviepy`
* `Pillow`
* `gtts`
* `yt_dlp`
* `dotenv`
* `requests`
* `huggingface_hub`
* `elevenlabs`
* `gradio_client`

---

## ⚙️ Variables de entorno

Crear un archivo `.env` con tus claves:

```env
HF_TOKEN=tu_token_de_huggingface
DEEPSEEK_API_KEY=tu_token_de_deepseek
ELEVENLABS_API_KEY=tu_token_de_elevenlabs
```

---

## 🚀 Uso

### Comando general

```bash
python main.py --ideas-file general-history.json
```

### Argumentos disponibles

| Flag                  | Descripción                                                                 |
| --------------------- | --------------------------------------------------------------------------- |
| `--ideas-file`        | Archivo JSON con ideas base (`ideas/*.json`)                                |
| `--mode`              | Modo de ejecución: `imagenes`, `audios`, `musica`, `video`, `juntar-audios` |
| `--story-id`          | Usar historia ya generada desde `stories/{id}`                              |
| `--subtitle-as-image` | Usar subtítulos como imágenes (mejor para evitar problemas con fuentes)     |

---

## 📁 Estructura de salida

Cada historia se guarda en una subcarpeta dentro de `stories/`:

```
stories/
└── 1a2b3c4d/
    ├── images/
    ├── audios/
    ├── story.json
    └── video.mp4
```

---

## 🧐 Formato de ideas

```json
[
  {
    "titulo": "La manzana de Newton",
    "descripcion": "Un día cualquiera, Isaac Newton observó una manzana caer..."
  }
]
```

---

## 🎨 Ejemplo de video generado

Puedes encontrar videos de ejemplo dentro del directorio `stories/` una vez ejecutado el script con éxito.

---

## ⚡ Características destacadas

* ✅ Imágenes generadas con estilo histórico y cinematográfico.
* ✅ Subtítulos personalizables como texto o imagen.
* ✅ Animaciones opcionales con modelo WAN2-1-FAST.
* ✅ Integración con ElevenLabs para voz profesional.
* ✅ Reintentos automáticos en caso de errores.

---

## ✊ Contribuciones

Este proyecto es un trabajo en curso. Sugerencias y mejoras son bienvenidas ❤️

---

## 🚫 Limitaciones

* Las animaciones pueden fallar ocasionalmente debido a la latencia del modelo.
* Se requiere configuración previa de `ffmpeg`, `convert` (ImageMagick), y las APIs.

---

## 🚀 Roadmap futuro

* [ ] Publicación automática en TikTok o YouTube Shorts
* [ ] Traducción automática multilingüe
* [ ] GUI amigable con Gradio o Streamlit

---

## 📅 Licencia

MIT License. Ver `LICENSE` para más detalles.

---

¡Gracias por usar **ReelGenerator**! 🌟
