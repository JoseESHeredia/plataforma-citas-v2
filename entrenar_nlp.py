# entrenar_nlp.py
import spacy
from spacy.training import Example
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
import random
import gspread
from google.oauth2.service_account import Credentials
import re
import os # ⭐️ Añadido para la lógica de HF
import json # ⭐️ Añadido para la lógica de HF


# --- Configuración ---
NOMBRE_DOCUMENTO = "Base de Datos Citas (Proyecto Voz y Chat)"
NOMBRE_HOJA_DATASET = "Dataset_Intenciones"
CARPETA_MODELO_GUARDADO = "modelo_intent_spacy"
INTENTS_VALIDOS = ["agendar", "consultar", "cancelar"]

# =========================================================================
# ⚠️ ADVERTENCIA: La lista EJEMPLOS_LOCALES usa claves diferentes a INTENTS_VALIDOS.
# Las claves deben ser 'agendar', 'consultar', 'cancelar' para que spaCy las reconozca.
# =========================================================================

def adaptar_ejemplos_locales(ejemplos_locales_brutos):
    """Adapta las claves de las categorías de la lista local a las requeridas."""
    datos_adaptados = []
    mapeo = {
        'AGENDAR_CITA': 'agendar',
        'CONSULTAR_CITA': 'consultar',
        'CANCELAR_CITA': 'cancelar'
    }
    
    for frase, data in ejemplos_locales_brutos:
        cats_nuevas = {}
        for clave_bruta, valor in data['cats'].items():
            if clave_bruta in mapeo:
                cats_nuevas[mapeo[clave_bruta]] = valor
        # Asegurarse de que solo se utilicen intenciones válidas
        datos_adaptados.append((frase, {'cats': cats_nuevas}))
    return datos_adaptados

EJEMPLOS_LOCALES = [
    # -------------------------------------------------------------------------
    # 🩺 AGENDAR CITA (AGENDAR_CITA: 1.0)
    # -------------------------------------------------------------------------
    
    # Frases Generales de Agendar
    ("quiero agendar una cita médica", {"cats": {"AGENDAR_CITA": 1.0, "CONSULTAR_CITA": 0.0, "CANCELAR_CITA": 0.0}}),
    ("necesito sacar una cita", {"cats": {"AGENDAR_CITA": 1.0, "CONSULTAR_CITA": 0.0, "CANCELAR_CITA": 0.0}}),
    ("deseo programar una consulta", {"cats": {"AGENDAR_CITA": 1.0, "CONSULTAR_CITA": 0.0, "CANCELAR_CITA": 0.0}}),
    ("me gustaría pedir una hora con un médico", {"cats": {"AGENDAR_CITA": 1.0, "CONSULTAR_CITA": 0.0, "CANCELAR_CITA": 0.0}}),
    ("dame una cita para la próxima semana", {"cats": {"AGENDAR_CITA": 1.0, "CONSULTAR_CITA": 0.0, "CANCELAR_CITA": 0.0}}),
    ("reservar una consulta lo antes posible", {"cats": {"AGENDAR_CITA": 1.0, "CONSULTAR_CITA": 0.0, "CANCELAR_CITA": 0.0}}),
    ("solicito una consulta médica", {"cats": {"AGENDAR_CITA": 1.0, "CONSULTAR_CITA": 0.0, "CANCELAR_CITA": 0.0}}),
    
    # Agendar con Mención de Médico
    ("quiero agendar con el Dr. Vega", {"cats": {"AGENDAR_CITA": 1.0, "CONSULTAR_CITA": 0.0, "CANCELAR_CITA": 0.0}}),
    ("necesito una cita con el doctor Pérez", {"cats": {"AGENDAR_CITA": 1.0, "CONSULTAR_CITA": 0.0, "CANCELAR_CITA": 0.0}}),
    ("puedo sacar hora con la Dra. Morales", {"cats": {"AGENDAR_CITA": 1.0, "CONSULTAR_CITA": 0.0, "CANCELAR_CITA": 0.0}}),
    ("deseo pedir una cita con el Dr. Castro", {"cats": {"AGENDAR_CITA": 1.0, "CONSULTAR_CITA": 0.0, "CANCELAR_CITA": 0.0}}),
    ("quiero programar una consulta con la Dra. Paredes", {"cats": {"AGENDAR_CITA": 1.0, "CONSULTAR_CITA": 0.0, "CANCELAR_CITA": 0.0}}),
    ("citas para el Dr. Vega", {"cats": {"AGENDAR_CITA": 1.0, "CONSULTAR_CITA": 0.0, "CANCELAR_CITA": 0.0}}),
    ("solicitar turno con el Dr. Pérez", {"cats": {"AGENDAR_CITA": 1.0, "CONSULTAR_CITA": 0.0, "CANCELAR_CITA": 0.0}}),
    ("hay disponibilidad con la Dra. Morales para consulta", {"cats": {"AGENDAR_CITA": 1.0, "CONSULTAR_CITA": 0.0, "CANCELAR_CITA": 0.0}}),

    # -------------------------------------------------------------------------
    # 🗓️ CONSULTAR CITA (CONSULTAR_CITA: 1.0)
    # -------------------------------------------------------------------------

    # Frases Generales de Consultar
    ("quiero consultar mis citas", {"cats": {"AGENDAR_CITA": 0.0, "CONSULTAR_CITA": 1.0, "CANCELAR_CITA": 0.0}}),
    ("cuándo tengo mi próxima cita", {"cats": {"AGENDAR_CITA": 0.0, "CONSULTAR_CITA": 1.0, "CANCELAR_CITA": 0.0}}),
    ("necesito ver mi historial de citas", {"cats": {"AGENDAR_CITA": 0.0, "CONSULTAR_CITA": 1.0, "CANCELAR_CITA": 0.0}}),
    ("saber el día y hora de mi consulta", {"cats": {"AGENDAR_CITA": 0.0, "CONSULTAR_CITA": 1.0, "CANCELAR_CITA": 0.0}}),
    ("revisar mis citas pendientes", {"cats": {"AGENDAR_CITA": 0.0, "CONSULTAR_CITA": 1.0, "CANCELAR_CITA": 0.0}}),
    ("dónde es mi cita de mañana", {"cats": {"AGENDAR_CITA": 0.0, "CONSULTAR_CITA": 1.0, "CANCELAR_CITA": 0.0}}),
    ("recuérdame mis turnos programados", {"cats": {"AGENDAR_CITA": 0.0, "CONSULTAR_CITA": 1.0, "CANCELAR_CITA": 0.0}}),
    
    # Consultar con Mención de Médico
    ("cuándo tengo cita con el Dr. Castro", {"cats": {"AGENDAR_CITA": 0.0, "CONSULTAR_CITA": 1.0, "CANCELAR_CITA": 0.0}}),
    ("quiero saber mi hora con la Dra. Paredes", {"cats": {"AGENDAR_CITA": 0.0, "CONSULTAR_CITA": 1.0, "CANCELAR_CITA": 0.0}}),
    ("tengo una consulta con el Dr. Vega para el jueves", {"cats": {"AGENDAR_CITA": 0.0, "CONSULTAR_CITA": 1.0, "CANCELAR_CITA": 0.0}}),
    ("ver la cita con la Dra. Morales", {"cats": {"AGENDAR_CITA": 0.0, "CONSULTAR_CITA": 1.0, "CANCELAR_CITA": 0.0}}),
    ("qué día me toca con el Dr. Pérez", {"cats": {"AGENDAR_CITA": 0.0, "CONSULTAR_CITA": 1.0, "CANCELAR_CITA": 0.0}}),
    
    # ⭐️ NUEVOS EJEMPLOS PARA 'CONSULTAR' (PARA EVITAR CONFUSIÓN)
    ("quiero ver mis citas", {"cats": {"AGENDAR_CITA": 0.0, "CONSULTAR_CITA": 1.0, "CANCELAR_CITA": 0.0}}),
    ("revisar mis horarios", {"cats": {"AGENDAR_CITA": 0.0, "CONSULTAR_CITA": 1.0, "CANCELAR_CITA": 0.0}}),
    ("dime mis citas", {"cats": {"AGENDAR_CITA": 0.0, "CONSULTAR_CITA": 1.0, "CANCELAR_CITA": 0.0}}),
    ("consultar mis horarios", {"cats": {"AGENDAR_CITA": 0.0, "CONSULTAR_CITA": 1.0, "CANCELAR_CITA": 0.0}}),
    ("consultar horarios", {"cats": {"AGENDAR_CITA": 0.0, "CONSULTAR_CITA": 1.0, "CANCELAR_CITA": 0.0}}),
    ("consultar tus horarios", {"cats": {"AGENDAR_CITA": 0.0, "CONSULTAR_CITA": 1.0, "CANCELAR_CITA": 0.0}}), # Tu ejemplo


    # -------------------------------------------------------------------------
    # ❌ CANCELAR CITA (CANCELAR_CITA: 1.0)
    # -------------------------------------------------------------------------

    # Frases Generales de Cancelar
    ("quiero cancelar mi cita de hoy", {"cats": {"AGENDAR_CITA": 0.0, "CONSULTAR_CITA": 0.0, "CANCELAR_CITA": 1.0}}),
    ("necesito anular mi consulta", {"cats": {"AGENDAR_CITA": 0.0, "CONSULTAR_CITA": 0.0, "CANCELAR_CITA": 1.0}}),
    ("deseo eliminar el turno", {"cats": {"AGENDAR_CITA": 0.0, "CONSULTAR_CITA": 0.0, "CANCELAR_CITA": 1.0}}),
    ("dar de baja una cita", {"cats": {"AGENDAR_CITA": 0.0, "CONSULTAR_CITA": 0.0, "CANCELAR_CITA": 1.0}}),
    ("anular la cita que tengo programada", {"cats": {"AGENDAR_CITA": 0.0, "CONSULTAR_CITA": 0.0, "CANCELAR_CITA": 1.0}}),
    ("ya no quiero la consulta del martes", {"cats": {"AGENDAR_CITA": 0.0, "CONSULTAR_CITA": 0.0, "CANCELAR_CITA": 1.0}}),
    
    # Cancelar con Mención de Médico
    ("quiero cancelar la cita con el Dr. Pérez", {"cats": {"AGENDAR_CITA": 0.0, "CONSULTAR_CITA": 0.0, "CANCELAR_CITA": 1.0}}),
    ("necesito anular mi hora con la Dra. Morales", {"cats": {"AGENDAR_CITA": 0.0, "CONSULTAR_CITA": 0.0, "CANCELAR_CITA": 1.0}}),
    ("cancelar la consulta con el Dr. Vega del viernes", {"cats": {"AGENDAR_CITA": 0.0, "CONSULTAR_CITA": 0.0, "CANCELAR_CITA": 1.0}}),
    ("ya no asistiré con el Dr. Castro", {"cats": {"AGENDAR_CITA": 0.0, "CONSULTAR_CITA": 0.0, "CANCELAR_CITA": 1.0}}),
    ("deseo cancelar el turno con la Dra. Paredes", {"cats": {"AGENDAR_CITA": 0.0, "CONSULTAR_CITA": 0.0, "CANCELAR_CITA": 1.0}}),

    # ⭐️ NUEVOS EJEMPLOS PARA 'CANCELAR' (PARA EVITAR CONFUSIÓN)
    ("quiero anular una cita", {"cats": {"AGENDAR_CITA": 0.0, "CONSULTAR_CITA": 0.0, "CANCELAR_CITA": 1.0}}),
    ("borrar mi cita", {"cats": {"AGENDAR_CITA": 0.0, "CONSULTAR_CITA": 0.0, "CANCELAR_CITA": 1.0}}),
    ("ya no iré", {"cats": {"AGENDAR_CITA": 0.0, "CONSULTAR_CITA": 0.0, "CANCELAR_CITA": 1.0}}),
    ("cancelar la cita", {"cats": {"AGENDAR_CITA": 0.0, "CONSULTAR_CITA": 0.0, "CANCELAR_CITA": 1.0}}),
    ("quiero cancelar", {"cats": {"AGENDAR_CITA": 0.0, "CONSULTAR_CITA": 0.0, "CANCELAR_CITA": 1.0}}),

]
# --- 1. Cargar y Preparar Datos (DESDE GOOGLE SHEETS) ---
def cargar_y_preparar_datos_gsheets():
    """Carga el dataset desde Google Sheets y lo prepara."""
    print(f"📥 Cargando datos desde Google Sheets: Hoja '{NOMBRE_HOJA_DATASET}'...")
    try:
        alcances = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
        
        # --- ⭐️ LÓGICA FUSIONADA (Compatible con HF y Local) ---
        google_creds_json = os.environ.get('GOOGLE_CREDENTIALS_JSON')
        if not google_creds_json:
            print("entrenar_nlp: Secret no encontrado, usando credenciales.json local...")
            cred = Credentials.from_service_account_file("credenciales.json", scopes=alcances)
        else:
            print("entrenar_nlp: Cargando credenciales desde Secret...")
            cred_dict = json.loads(google_creds_json)
            cred = Credentials.from_service_account_info(cred_dict, scopes=alcances)
        
        cliente = gspread.authorize(cred)
        # --- Fin de Lógica Fusionada ---
        
        documento = cliente.open(NOMBRE_DOCUMENTO)
        hoja = documento.worksheet(NOMBRE_HOJA_DATASET)
        datos = hoja.get_all_records()
        if not datos: raise ValueError("La hoja del dataset está vacía.")
        df = pd.DataFrame(datos)
        print(f"✅ Datos cargados desde GSheets: {len(df)} filas.")

        if not all(col in df.columns for col in ['frase', 'intent']):
            raise ValueError("Faltan las columnas 'frase' o 'intent'")
        df = df.dropna(subset=['frase', 'intent'])

        def preprocesar_texto(s: str) -> str:
             s = str(s).lower().strip(); s = re.sub(r"\s+", " ", s); return s
        df['frase_limpia'] = df['frase'].apply(preprocesar_texto)

        df = df[df['intent'].isin(INTENTS_VALIDOS)]
        if df.empty: raise ValueError("Dataset vacío después de limpiar/filtrar.")
        print(f"✅ Dataset limpiado y filtrado: {len(df)} filas.")
        print("Distribución de intenciones:"); print(df['intent'].value_counts())

        datos_spacy = []
        for _, row in df.iterrows():
            cats = {intent: (intent == row['intent']) for intent in INTENTS_VALIDOS}
            datos_spacy.append((row['frase_limpia'], {'cats': cats}))
        return datos_spacy
    except FileNotFoundError: print("❌ Error: No se encontró 'credenciales.json'."); return None
    except Exception as e: print(f"❌ Error al cargar/preparar datos GSheets: {e}"); return None

# --- 2. Entrenar Modelo spaCy textcat (CORREGIDO) ---
def entrenar_modelo_spacy(datos_entrenamiento, carpeta_salida):
    """Entrena un modelo textcat de spaCy."""
    print("🧠 Iniciando entrenamiento del modelo spaCy...")

    nlp = spacy.blank("es")  # modelo base en blanco
    if "textcat" not in nlp.pipe_names:
        textcat = nlp.add_pipe("textcat")
    else:
        textcat = nlp.get_pipe("textcat")

    # Añadir las etiquetas (intenciones válidas)
    for intent in INTENTS_VALIDOS:
        textcat.add_label(intent)

    # Definir hiperparámetros de entrenamiento
    n_iter = 10               # 🔁 Número de épocas (puedes ajustar entre 5–20)
    batch_sizes = [8, 16, 32] # 📦 Tamaños de lote para minibatch

    # Inicializar el optimizador
    optimizer = nlp.initialize()

    print("Iniciando bucle de entrenamiento...")
    for i in range(n_iter):
        random.shuffle(datos_entrenamiento)
        from spacy.util import compounding
        batches = spacy.util.minibatch(datos_entrenamiento, size=compounding(4.0, 32.0, 1.5))
        losses = {}
        batch_count = 0
        for batch in batches:
            texts, annotations = zip(*batch)
            examples = [Example.from_dict(nlp.make_doc(t), a) for t, a in zip(texts, annotations)]

            try:
                nlp.update(examples, sgd=optimizer, drop=0.2, losses=losses)
                batch_count += 1
            except Exception as e_update:
                print(f"❌ Error durante nlp.update en batch {batch_count}: {e_update}")
        print(f"  Época {i+1}/{n_iter}, Pérdida: {losses.get('textcat', 0.0):.3f}")

    # Guardar modelo
    try:
        nlp.to_disk(carpeta_salida)
        print(f"\n💾 Modelo spaCy entrenado guardado en: {carpeta_salida}")
        return nlp
    except Exception as e_save:
        print(f"❌ Error al guardar el modelo: {e_save}")
        return None



# --- 3. Evaluar Modelo (Sin cambios) ---
def evaluar_modelo(nlp_modelo, datos_prueba):
    print("\n📊 Evaluando modelo...")
    textos_prueba, anotaciones_prueba = zip(*datos_prueba)
    predicciones = []
    reales = []
    for texto, anotacion in zip(textos_prueba, anotaciones_prueba):
        doc = nlp_modelo(texto)
        prediccion = max(doc.cats, key=doc.cats.get)
        predicciones.append(prediccion)
        real = [intent for intent, valor in anotacion['cats'].items() if valor][0]
        reales.append(real)

    f1_macro = f1_score(reales, predicciones, average='macro', zero_division=0)
    print(f"\n  F1-Score (macro): {f1_macro:.4f}")
    print("\n  Reporte de Clasificación Detallado:")
    print(classification_report(reales, predicciones, zero_division=0))

    if f1_macro >= 0.90: print("\n🎉 ¡Meta cumplida! F1-Score >= 0.90")
    else: print("\n⚠️ F1-Score < 0.90. El modelo necesita mejorar.")

# --- Ejecución Principal ---
# --- Ejecución Principal ---
if __name__ == "__main__":
    datos_completos = cargar_y_preparar_datos_gsheets()

    # Si no se pudieron cargar los datos desde GSheets, usa ejemplos locales
    if not datos_completos:
        print("⚠️ No se pudo cargar dataset desde Google Sheets. Usando ejemplos locales por defecto...")
        # ⭐️ INICIO DE LA CORRECCIÓN: Bug tipográfico
        datos_completos = adaptar_ejemplos_locales(EJEMPLOS_LOCALES) # ✅ Corregido de 'datos_comleto'
        # ⭐️ FIN DE LA CORRECCIÓN

    if datos_completos:
        train_data, test_data = train_test_split(datos_completos, test_size=0.2, random_state=42)
        print(f"\nDatos divididos: {len(train_data)} para entrenar, {len(test_data)} para probar.")
        modelo_entrenado = entrenar_modelo_spacy(train_data, CARPETA_MODELO_GUARDADO)
        if modelo_entrenado and test_data:
            evaluar_modelo(modelo_entrenado, test_data)
        else:
            print("❌ No se pudo entrenar o evaluar el modelo.")
