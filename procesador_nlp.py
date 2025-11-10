import spacy
import re
from datetime import datetime, timedelta

# --- Cargar Modelo Entrenado (Tarea S2-02 REAL) ---
MODELO_INTENT_PATH = "modelo_intent_spacy" # Carpeta donde guardó entrenar_nlp.py
try:
    # Intenta cargar el modelo de intenciones
    nlp_intent = spacy.load(MODELO_INTENT_PATH)
    print(f"✅ Modelo NLP de intenciones cargado desde: {MODELO_INTENT_PATH}")
    modelo_cargado = True
except IOError:
    print(f"❌ Error: No se pudo cargar el modelo de intenciones desde '{MODELO_INTENT_PATH}'.")
    print("   Asegúrate de que el archivo existe y el entrenamiento fue exitoso.")
    nlp_intent = None
    modelo_cargado = False
except Exception as e:
    print(f"❌ Error inesperado al cargar modelo de intenciones: {e}")
    nlp_intent = None
    modelo_cargado = False


# --- Cargar Modelo Base (Para Entidades - S2-03) ---
# Necesario para extraer entidades como PER
try:
    # Usamos el modelo base 'es_core_news_sm'
    nlp_base = spacy.load("es_core_news_sm")
    print("✅ Modelo base spaCy 'es_core_news_sm' (para entidades) cargado.")
except IOError:
    print("❌ Error: Modelo base 'es_core_news_sm' no encontrado.")
    print("   Asegúrate de que 'setup.sh' descargó el modelo.")
    nlp_base = None # No podremos extraer entidades si falla


# --- Detección de Intenciones (Usando Modelo) ---
def detectar_intencion_modelo(texto):
    """
    Usa el modelo spaCy textcat entrenado para predecir la intención.
    """
    if not modelo_cargado or not nlp_intent:
        print("Advertencia: Modelo de intenciones no cargado. Usando fallback 'desconocido'.")
        return "desconocido" # Fallback si el modelo no cargó

    # Preprocesar texto (igual que en el entrenamiento)
    texto_limpio = str(texto).lower().strip()
    texto_limpio = re.sub(r"\s+", " ", texto_limpio)

    # Predecir con el modelo cargado
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
        return {} # No se puede procesar si spaCy base no cargó

    doc = nlp_base(texto) # Usa el modelo base pre-entrenado
    entidades = {}

    # 1. Extraer Médico (NER Persona)
    for ent in doc.ents:
        if ent.label_ == "PER":
            # Verificamos si la persona es un médico conocido
            if any(medico in ent.text for medico in ["Vega", "Perez", "Morales", "Castro", "Paredes"]):
                entidades["medico"] = ent.text 
                break

    # 2. Extracer DNI (Regex)
    match_dni = re.search(r'\b(\d{8})\b', texto)
    if match_dni:
        # ⭐️⭐️⭐️ AQUÍ ESTÁ LA CORRECCIÓN ⭐️⭐️⭐️
        entidades["DNI"] = match_dni.group(1) # Cambiado de 'dni' a 'DNI'

    # 3. Extraer Fecha (Reglas simples)
    texto_lower = texto.lower()
    if "mañana" in texto_lower:
        entidades["fecha"] = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
    elif "hoy" in texto_lower:
        entidades["fecha"] = datetime.now().strftime("%Y-%m-%d")
    else:
        # Regex simple para AAAA-MM-DD
        match_fecha_iso = re.search(r'(\d{4}-\d{2}-\d{2})', texto)
        if match_fecha_iso:
             entidades["fecha"] = match_fecha_iso.group(1)

    # 4. Extraer Hora (Reglas simples)
    match_hora = re.search(r'(\d{1,2}:\d{2})\s*(am|pm)?', texto_lower)
    if match_hora:
        hora_str = match_hora.group(1)
        partes = hora_str.split(':')
        if len(partes) == 2:
             entidades["hora"] = f"{int(partes[0]):02d}:{int(partes[1]):02d}"
    elif " a las " in texto_lower:
        partes = texto_lower.split(" a las ")
        if len(partes) > 1:
            hora_potencial = partes[1].split()[0].replace(':','').replace('h','') 
            if hora_potencial.isdigit():
                 hora_num = int(hora_potencial)
                 if 0 <= hora_num <= 23: 
                      entidades["hora"] = f"{hora_num:02d}:00"


    return entidades

# --- Función Principal ---
def procesar_texto(texto):
    """
    Combina detección de intención (con modelo) y extracción de entidades.
    """
    intencion = detectar_intencion_modelo(texto)
    entidades = extraer_entidades(texto)

    return intencion, entidades
