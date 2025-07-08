# 🎮 ReelGenerator: Automated Video Stories

This project generates **short vertical-style videos** ("reels", 2 minutes long) based on real or fictional stories by combining **narration, AI-generated images, voiceovers, background music, and subtitles**.

---

## ✨ What Does This Project Do?

Given a JSON file with story ideas (`ideas/*.json`), the system:

1. Generates a structured script (text + images + music) using DeepSeek AI.
2. Creates realistic visuals aligned with the narrative using Hugging Face models.
3. Narrates the text with ElevenLabs or gTTS.
4. Adds matching instrumental background music.
5. Produces a final video with cinematic transitions, zoom, motion, and subtitles.

---

## 🛠️ Requirements

### Python

* Python 3.9 or higher

### Libraries

Install dependencies with:

```bash
pip install -r requirements.txt
```

Includes:

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

## ⚙️ Environment Variables

Create a `.env` file with the following keys:

```env
HF_TOKEN=your_huggingface_token
DEEPSEEK_API_KEY=your_deepseek_token
ELEVENLABS_API_KEY=your_elevenlabs_token
```

---

## 🚀 Usage

### Basic command

```bash
python main.py --ideas-file general-history.json
```

### Available arguments

| Flag                  | Description                                                              |
| --------------------- | ------------------------------------------------------------------------ |
| `--ideas-file`        | Path to the input JSON file (`ideas/*.json`)                             |
| `--mode`              | Execution mode: `imagenes`, `audios`, `musica`, `video`, `juntar-audios` |
| `--story-id`          | Use an already generated story from `stories/{id}`                       |
| `--subtitle-as-image` | Render subtitles as images (useful to avoid font rendering issues)       |

---

## 📁 Output Structure

Each story is saved under its own folder in `stories/`:

```
stories/
└── 1a2b3c4d/
    ├── images/
    ├── audios/
    ├── story.json
    └── video.mp4
```

---

## 🧐 Story Format Example

```json
[
  {
    "titulo": "Newton's Apple",
    "descripcion": "One ordinary day, Isaac Newton watched an apple fall..."
  }
]
```

---

## 🎨 Sample Output

You can find sample videos under the `stories/` directory after running the script.

---

## ⚡ Key Features

* ✅ Historically inspired cinematic visuals
* ✅ Subtitles as text or image overlays
* ✅ Optional animated scenes using WAN2-1-FAST
* ✅ ElevenLabs integration for professional voiceovers
* ✅ Automatic retries on generation errors

---

## ✊ Contributing

This project is a work in progress. Feedback and contributions are welcome ❤️

---

## 🚫 Limitations

* Animation may occasionally fail due to model latency.
* Requires preinstalled tools: `ffmpeg`, `convert` (ImageMagick), and valid API keys.

---

## 🚀 Future Roadmap

* [ ] Automatic publishing to TikTok or YouTube Shorts
* [ ] Multi-language translation support
* [ ] User-friendly UI via Gradio or Streamlit

---

## 📅 License

MIT License. See `LICENSE` for details.

---

Thanks for using **ReelGenerator**! 🌟
