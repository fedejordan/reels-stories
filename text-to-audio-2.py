from gtts import gTTS
import os

texto = "Hola, esto es una prueba de texto a voz usando Google Text to Speech"
tts = gTTS(text=texto, lang='es')
tts.save("output.mp3")

# Reproducir (opcional)
# os.system("start output.mp3")  # Windows
os.system("afplay output.mp3")  # Mac
# os.system("mpg123 output.mp3")  # Linux
