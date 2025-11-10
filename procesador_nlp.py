import spacy
import re
import dateparser # ⭐️ AÑADIDO (Mejora Sprint 3)
from datetime import datetime, timedelta

# --- Cargar Modelo Entrenado (Tarea S2-02 REAL) ---
MODELO_INTENT_PATH = "modelo_intent_spacy" # Carpeta donde guardó entrenar_nlp.py
try:
    nlp_intent = spacy.load(MODELO_INTENT_PATH)
    print(f"✅ Modelo NLP de intenciones cargado desde: {MODELO_INTENT_PATH}")
    modelo_cargado = True
except IOError:
    print(f"❌ Error: No se pudo cargar el modelo de intenciones desde '{MODELO_INTENT_PATH}'.")
    nlp_intent = None
    modelo_cargado = False
except Exception as e:
    print(f"❌ Error inesperado al cargar modelo de intenciones: {e}")
    nlp_intent = None
    modelo_cargado = False


# --- Cargar Modelo Base (Para Entidades - S2-03) ---
try:
    nlp_base = spacy.load("es_core_news_sm")
    print("✅ Modelo base spaCy 'es_core_news_sm' (para entidades) cargado.")
except IOError:
    print("❌ Error: Modelo base 'es_core_news_sm' no encontrado.")
    nlp_base = None # No podremos extraer entidades si falla


# --- Detección de Intenciones (Usando Modelo) ---
def detectar_intencion_modelo(texto):
    """
    Usa el modelo spaCy textcat entrenado para predecir la intención.
    """
    if not modelo_cargado or not nlp_intent:
        return "desconocido" 

    texto_limpio = str(texto).lower().strip()
    texto_limpio = re.sub(r"\s+", " ", texto_limpio)
    doc = nlp_intent(texto_limpio)
    intencion_predicha = max(doc.cats, key=doc.cats.get)
    score = doc.cats[intencion_predicha]
    print(f"  Predicción de Intención: {intencion_predicha} (Score: {score:.2f})")

    return intencion_predicha


# --- Extractor de Entidades ---
def extraer_entidades(texto):
    """
    Extrae entidades como DNI, Fecha y Médico (usa nlp_base).
    """
    if not nlp_base:
        print("Advertencia: Modelo base no cargado. No se pueden extraer entidades.")
        return {} 

    doc = nlp_base(texto) 
    entidades = {}
    texto_lower = texto.lower()

    # 1. Extraer Médico (NER Persona)
    for ent in doc.ents:
        if ent.label_ == "PER":
            # Verificamos si la persona es un médico conocido
            if any(medico in ent.text for medico in ["Vega", "Perez", "Morales", "Castro", "Paredes"]):
                entidades["Medico"] = ent.text # ⭐️ CAMBIO a Mayúscula
                break

    # 2. Extracer DNI (Regex)
    match_dni = re.search(r'\b(\d{8})\b', texto)
    if match_dni:
        entidades["DNI"] = match_dni.group(1) # ⭐️ CAMBIO a Mayúscula

    # 3. Extraer Fecha (Usando Dateparser) - ⭐️ MEJORA SPRINT 3
    # Configuraciones para entender "mañana", "próximo martes", "10-11-2025"
    settings = {'PREFER_DATES_FROM': 'future', 'DATE_ORDER': 'DMY'}
    # Quitamos las palabras de hora para que dateparser no se confunda
    texto_sin_hora = re.sub(r'(\d{1,2})\s*(?::(\d{2}))?\s*(am|pm)?', '', texto_lower)
    texto_sin_hora = re.sub(r'a las \d+', '', texto_sin_hora)

    fecha_obj = dateparser.parse(texto_sin_hora, settings=settings)
    if fecha_obj:
        entidades["Fecha"] = fecha_obj.strftime("%Y-%m-%d") # ⭐️ CAMBIO a Mayúscula
    else:
        # Regex simple para AAAA-MM-DD (fallback)
        match_fecha_iso = re.search(r'(\d{4}-\d{2}-\d{2})', texto)
        if match_fecha_iso:
             entidades["Fecha"] = match_fecha_iso.group(1) # ⭐️ CAMBIO a Mayúscula

    # 4. Extraer Hora (Reglas simples) - ⭐️ CORREGIDO (Bug G)
    # Busca formatos como "14:30", "2pm", "3 pm", "a las 15"
    match_hora = re.search(r'(\b\d{1,2})\s*(?::(\d{2}))?\s*(am|pm)?', texto_lower)
    
    if match_hora:
        hora_str = match_hora.group(1)
        min_str = match_hora.group(2) or "00" # Minutos (default a "00")
        am_pm = match_hora.group(3)

        try:
            hora_num = int(hora_str)
            min_num = int(min_str)
            
            # Convertir a 24h si hay 'pm'
            if am_pm == 'pm' and hora_num < 12:
                hora_num += 12
            # Corregir 12am (medianoche)
            elif am_pm == 'am' and hora_num == 12:
                hora_num = 0

            if 0 <= hora_num <= 23 and 0 <= min_num <= 59:
                entidades["Hora"] = f"{hora_num:02d}:{min_num:02d}" # ⭐️ CAMBIO a Mayúscula
                
        except ValueError:
            pass # Ignorar si no es un número válido

    # Fallback para "a las 2" (si el regex de arriba falla)
    elif " a las " in texto_lower and "Hora" not in entidades:
        partes = texto_lower.split(" a las ")
        if len(partes) > 1:
            hora_potencial = partes[1].split()[0].replace('h','') 
            if hora_potencial.isdigit():
                 hora_num = int(hora_potencial)
                 if 0 <= hora_num <= 23: 
                      entidades["Hora"] = f"{hora_num:02d}:00" # ⭐️ CAMBIO a Mayúscula

    return entidades

# --- Función Principal ---
def procesar_texto(texto):
    """
    Combina detección de intención (con modelo) y extracción de entidades.
    """
    intencion = detectar_intencion_modelo(texto)
    entidades = extraer_entidades(texto)

    return intencion, entidades
