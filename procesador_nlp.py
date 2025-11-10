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


# ⭐️ Lista de Médicos conocida para look-up más robusto (FIX)
MEDICOS_CONOCIDOS = ["Vega", "Perez", "Morales", "Castro", "Paredes"]

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

    # 1. Extraer Médico (NER Persona + Look-up más robusto - FIX)
    medico_encontrado = None
    for ent in doc.ents:
        if ent.label_ == "PER":
            # Verificamos si la persona es un médico conocido
            if any(medico in ent.text for medico in MEDICOS_CONOCIDOS):
                medico_encontrado = ent.text 
                break
    
    # Fallback: Buscar los nombres conocidos en el texto plano
    if not medico_encontrado:
        for medico in MEDICOS_CONOCIDOS:
            # Buscamos 'dr. vega', 'dra. morales', o 'perez'
            if re.search(r'\b(dr|dra|doctor|doctora)\.?\s*' + re.escape(medico) + r'\b', texto_lower) or re.search(r'\b' + re.escape(medico) + r'\b', texto_lower):
                 # Usamos el nombre limpio (ej. 'Dr.Perez') si lo encontramos
                 medico_encontrado = "Dr." + medico if medico != "Morales" and medico != "Paredes" else "Dra." + medico
                 break
    
    if medico_encontrado:
         entidades["Medico"] = medico_encontrado 


    # 2. Extracer DNI (Regex)
    # Busca 8 dígitos seguidos, con o sin puntos/espacios (que se limpiarán después)
    match_dni = re.search(r'\b(\d{1}\s*\d{3}\s*\d{4}|\d{8})\b', texto)
    if match_dni:
        # Aquí solo capturamos, la limpieza final de espacios/puntos la hace validar_formato
        entidades["DNI"] = match_dni.group(0).replace(' ', '') 

    # 3. Extraer Fecha (Usando Dateparser) - ⭐️ CORREGIDO (Bug 4)
    # Configuraciones para entender "mañana", "próximo martes", "10-11-2025"
    settings = {'PREFER_DATES_FROM': 'future', 'DATE_ORDER': 'DMY'}
    
    # Quitamos las palabras de hora para que dateparser no se confunda
    texto_sin_hora = re.sub(r'(\d{1,2})\s*(?::(\d{2}))?\s*(am|pm)?', '', texto_lower)
    texto_sin_hora = re.sub(r'a las \d+', '', texto_sin_hora)

    fecha_obj = dateparser.parse(texto_sin_hora, languages=['es'], settings=settings) 
    
    # ⭐️ FIX: Solo aceptamos fechas futuras.
    if fecha_obj and fecha_obj.date() >= datetime.today().date(): 
        entidades["Fecha"] = fecha_obj.strftime("%Y-%m-%d")
    
    # 4. Extraer Hora (Reglas simples) - ⭐️ CORREGIDO (Bug G)
    # Busca formatos como "14:30", "2pm", "3 pm", "a las 15h"
    match_hora = re.search(r'(\b\d{1,2})\s*(?::(\d{2}))?\s*(am|pm|h)?\b', texto_lower)
    
    if match_hora:
        hora_str = match_hora.group(1)
        min_str = match_hora.group(2) or "00" # Minutos (default a "00")
        am_pm = match_hora.group(3)
        h_suffix = match_hora.group(3) if match_hora.group(3) == 'h' else None # 'h' si existe

        try:
            hora_num = int(hora_str)
            min_num = int(min_str)
            
            # Convertir a 24h si hay 'pm'
            if am_pm == 'pm' and hora_num < 12:
                hora_num += 12
            # Corregir 12am (medianoche)
            elif am_pm == 'am' and hora_num == 12:
                hora_num = 0

            # Aceptar 15h (aunque sea redundante)
            if h_suffix == 'h' and hora_num > 12:
                 pass # Ya está en 24h

            if 0 <= hora_num <= 23 and 0 <= min_num <= 59:
                entidades["Hora"] = f"{hora_num:02d}:{min_num:02d}" 
                
        except ValueError:
            pass # Ignorar si no es un número válido

    # Fallback para "a las 2" (si el regex de arriba falla o es ambiguo)
    elif " a las " in texto_lower and "Hora" not in entidades:
        partes = texto_lower.split(" a las ")
        if len(partes) > 1:
            hora_potencial = partes[1].split()[0].replace('h','') 
            if hora_potencial.isdigit():
                 hora_num = int(hora_potencial)
                 if 0 <= hora_num <= 23: 
                      entidades["Hora"] = f"{hora_num:02d}:00" 

    return entidades

# --- Función Principal ---
def procesar_texto(texto):
    """
    Combina detección de intención (con modelo) y extracción de entidades.
    """
    intencion = detectar_intencion_modelo(texto)
    entidades = extraer_entidades(texto)

    return intencion, entidades
