import spacy
import re
from datetime import datetime, timedelta

# --- Cargar Modelo Entrenado (Tarea S2-02 REAL) ---
MODELO_INTENT_PATH = "modelo_intent_spacy" # Carpeta donde guardó entrenar_nlp.py
try:
    nlp_intent = spacy.load(MODELO_INTENT_PATH)
    print(f"✅ Modelo NLP de intenciones cargado desde: {MODELO_INTENT_PATH}")
    modelo_cargado = True
except IOError:
    print(f"❌ Error: No se pudo cargar el modelo de intenciones desde '{MODELO_INTENT_PATH}'.")
    print("   Asegúrate de haber ejecutado 'entrenar_nlp.py' exitosamente.")
    nlp_intent = None # Usar None si falla la carga
    modelo_cargado = False
except Exception as e:
    print(f"❌ Error inesperado al cargar modelo de intenciones: {e}")
    nlp_intent = None
    modelo_cargado = False


# --- Cargar Modelo Base (Para Entidades - S2-03) ---
# Seguimos necesitando un modelo base para extraer entidades como PER (Médico)
try:
    # Usamos el modelo pequeño que ya teníamos descargado
    nlp_base = spacy.load("es_core_news_sm")
    print("✅ Modelo base spaCy 'es_core_news_sm' (para entidades) cargado.")
except IOError:
    print("❌ Error: Modelo base 'es_core_news_sm' no encontrado.")
    print("   Ejecuta: python -m spacy download es_core_news_sm")
    nlp_base = None # No podremos extraer entidades si falla


# --- Tarea S2-02: Detección de Intenciones (Usando Modelo) ---
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
    # Obtener la categoría con la puntuación más alta
    intencion_predicha = max(doc.cats, key=doc.cats.get)
    score = doc.cats[intencion_predicha]
    print(f"  Predicción de Intención: {intencion_predicha} (Score: {score:.2f})")

    # (Opcional: Podrías añadir un umbral de confianza aquí si quisieras)
    # if score < 0.5: return "desconocido"

    return intencion_predicha


# --- Tarea S2-03: Extractor de Entidades (Sin cambios, usa nlp_base) ---
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
                entidades["medico"] = ent.text # Podríamos normalizar a "Dr. Perez" si quisiéramos
                break

    # 2. Extraer DNI (Regex)
    match_dni = re.search(r'\b(\d{8})\b', texto)
    if match_dni:
        entidades["dni"] = match_dni.group(1)

    # 3. Extraer Fecha (Reglas simples)
    texto_lower = texto.lower()
    if "mañana" in texto_lower:
        entidades["fecha"] = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
    elif "hoy" in texto_lower:
        entidades["fecha"] = datetime.now().strftime("%Y-%m-%d")
    # (Faltaría regex o Matcher para fechas explícitas como '2025-11-15')
    else:
        # Regex simple para AAAA-MM-DD
        match_fecha_iso = re.search(r'(\d{4}-\d{2}-\d{2})', texto)
        if match_fecha_iso:
             entidades["fecha"] = match_fecha_iso.group(1)

    # 4. Extraer Hora (Reglas simples)
    match_hora = re.search(r'(\d{1,2}:\d{2})\s*(am|pm)?', texto_lower)
    if match_hora:
        hora_str = match_hora.group(1)
        # Asegurarse de formato HH:MM (ej. 9:30 -> 09:30)
        partes = hora_str.split(':')
        if len(partes) == 2:
             entidades["hora"] = f"{int(partes[0]):02d}:{int(partes[1]):02d}"
    elif " a las " in texto_lower:
        partes = texto_lower.split(" a las ")
        if len(partes) > 1:
            hora_potencial = partes[1].split()[0].replace(':','').replace('h','') # Tomar palabra/número después de "a las", quitar ':' o 'h'
            if hora_potencial.isdigit():
                 hora_num = int(hora_potencial)
                 if 0 <= hora_num <= 23: # Asumir hora en punto si solo dan número
                      entidades["hora"] = f"{hora_num:02d}:00"


    return entidades

# --- Función Principal (Ahora llama al modelo) ---
def procesar_texto(texto):
    """
    Combina detección de intención (con modelo) y extracción de entidades.
    """
    # Usar la nueva función que carga el modelo entrenado
    intencion = detectar_intencion_modelo(texto)
    # La extracción de entidades sigue igual
    entidades = extraer_entidades(texto)

    return intencion, entidades


# ===== TEST AUTOMÁTICO (Ahora prueba el modelo cargado) =====
if __name__ == "__main__":
    print("\n📌 Iniciando test de NLP (Sprint 2 - Usando Modelo Entrenado)...\n")

    casos_prueba = [
        "Quiero agendar una cita con el Dr.Vega para mañana a las 10:00",
        "necesito ver mis citas con mi dni 12345678",
        "cancela mi cita para 2025-10-30 por favor", # Cambiada para incluir fecha
        "hola buenos días",
        "Necesito una cita, mi DNI es 98765432" # El caso que falló antes
    ]

    for texto in casos_prueba:
        print(f"--- Procesando: '{texto}' ---")
        intencion, entidades = procesar_texto(texto)
        print(f"  > Intención (Modelo): {intencion}")
        print(f"  > Entidades: {entidades}\n")

    print("\n✅ Test de NLP completado.")
