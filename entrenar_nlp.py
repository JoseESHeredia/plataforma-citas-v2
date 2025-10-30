# entrenar_nlp.py
import spacy
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
import random
import gspread
from google.oauth2.service_account import Credentials
import re
import os   # <-- 1. IMPORT AÑADIDO
import json # <-- 2. IMPORT AÑADIDO

# --- Configuración ---
NOMBRE_DOCUMENTO = "Base de Datos Citas (Proyecto Voz y Chat)"
NOMBRE_HOJA_DATASET = "Dataset_Intenciones"
CARPETA_MODELO_GUARDADO = "modelo_intent_spacy"
INTENTS_VALIDOS = ["agendar", "consultar", "cancelar"]

# --- 1. Cargar y Preparar Datos (DESDE GOOGLE SHEETS) ---
def cargar_y_preparar_datos_gsheets():
    """Carga el dataset desde Google Sheets y lo prepara."""
    print(f"📥 Cargando datos desde Google Sheets: Hoja '{NOMBRE_HOJA_DATASET}'...")
    try:
        # --- 3. BLOQUE DE SECRETS E INDENTACIÓN CORREGIDO ---
        alcances = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
        
        google_creds_json = os.environ.get('GOOGLE_CREDENTIALS_JSON')
        if not google_creds_json:
            print("Secret no encontrado, usando credenciales.json local...")
            cred = Credentials.from_service_account_file("credenciales.json", scopes=alcances)
        else:
            print("Cargando credenciales desde Secret...")
            cred_dict = json.loads(google_creds_json)
            cred = Credentials.from_service_account_info(cred_dict, scopes=alcances)
        
        # Estas líneas van DESPUÉS del if/else, pero DENTRO del try
        cliente = gspread.authorize(cred)
        documento = cliente.open(NOMBRE_DOCUMENTO)
        hoja = documento.worksheet(NOMBRE_HOJA_DATASET)
        datos = hoja.get_all_records()
        if not datos: raise ValueError("La hoja del dataset está vacía.")
        df = pd.DataFrame(datos)
        print(f"✅ Datos cargados desde GSheets: {len(df)} filas.")
        # --- FIN DE LA CORRECCIÓN DE INDENTACIÓN ---

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
    print("\n🧠 Iniciando entrenamiento del modelo spaCy textcat...")
    nlp = spacy.blank("es")
    if "textcat" not in nlp.pipe_names:
        textcat = nlp.add_pipe("textcat", last=True) # Usar config por defecto
        print("Usando configuración por defecto para 'textcat'.")
    else:
        textcat = nlp.get_pipe("textcat")
    for intent in INTENTS_VALIDOS: textcat.add_label(intent)

    n_iter = 10
    optimizer = nlp.begin_training()
    batch_sizes = spacy.util.compounding(4.0, 32.0, 1.001)
    print("Iniciando bucle de entrenamiento...")
    for i in range(n_iter):
        random.shuffle(datos_entrenamiento)
        batches = spacy.util.minibatch(datos_entrenamiento, size=batch_sizes)
        losses = {}
        batch_count = 0
        for batch in batches:
            texts, annotations = zip(*batch)
            try: nlp.update(texts, annotations, sgd=optimizer, drop=0.2, losses=losses)
            except Exception as e_update: print(f"❌ Error durante nlp.update: {e_update}")
        print(f"  Época {i+1}/{n_iter}, Pérdida: {losses.get('textcat', 0.0):.3f}")
    try:
        nlp.to_disk(carpeta_salida)
        print(f"\n💾 Modelo spaCy entrenado guardado en: {carpeta_salida}")
        return nlp
    except Exception as e_save: print(f"❌ Error al guardar el modelo: {e_save}"); return None

# --- 3. Evaluar Modelo (Sin cambios) ---
def evaluar_modelo(nlp_modelo, datos_prueba):
    print("\n📊 Evaluando modelo...")
    textos_prueba, anotaciones_prueba = zip(*datos_prueba)
    predicciones = []; reales = []
    for texto, anotacion in zip(textos_prueba, anotaciones_prueba):
        doc = nlp_modelo(texto)
        prediccion = max(doc.cats, key=doc.cats.get)
        predicciones.append(prediccion)
        real = [intent for intent, valor in anotacion['cats'].items() if valor][0]
        reales.append(real)

    f1_macro = f1_score(reales, predicciones, average='macro', zero_division=0)
    print(f"\n  F1-Score (macro): {f1_macro:.4f}")
    print("\n  Reporte de Clasificación Detallado:"); print(classification_report(reales, predicciones, zero_division=0))
    if f1_macro >= 0.90: print("\n🎉 ¡Meta cumplida! F1-Score >= 0.90")
    else: print("\n⚠️ F1-Score < 0.90. El modelo necesita mejorar.")

# --- Ejecución Principal ---
if __name__ == "__main__":
    datos_completos = cargar_y_preparar_datos_gsheets()
    if datos_completos:
        train_data, test_data = train_test_split(datos_completos, test_size=0.2, random_state=42)
        print(f"\nDatos divididos: {len(train_data)} para entrenar, {len(test_data)} para probar.")
        modelo_entrenado = entrenar_modelo_spacy(train_data, CARPETA_MODELO_GUARDADO)
        if modelo_entrenado and test_data:
            evaluar_modelo(modelo_entrenado, test_data)
        else: print("❌ No se pudo entrenar o evaluar el modelo.")
