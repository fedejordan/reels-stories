import pyttsx3

engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Velocidad de habla
engine.setProperty('volume', 1.0)  # Volumen (0.0 a 1.0)

texto = "Hola, esto es una prueba de texto a voz usando pyttsx3"
engine.say(texto)
engine.runAndWait()
