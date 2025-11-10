import time
try:
    from faster_whisper import WhisperModel
    model = WhisperModel("small", device="cpu")
    model_loaded = True
except ImportError:
    model_loaded = False

def transcribir_audio(ruta_audio):
    """Devuelve el texto transcrito del archivo."""
    if not model_loaded:
        return "⚠️ Transcripción no disponible. Instala 'faster-whisper' con: pip install faster-whisper"
    try:
        time.sleep(0.5)
        segments, info = model.transcribe(ruta_audio)
        texto = " ".join([seg.text for seg in segments])
        return texto.strip()
    except Exception as e:
        return f"❌ Error al transcribir: {e}"
