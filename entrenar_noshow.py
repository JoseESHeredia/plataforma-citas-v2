import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
import gspread
from google.oauth2.service_account import Credentials
import joblib # Para guardar el modelo
import os # ⭐️ Añadido para la lógica de HF
import json # ⭐️ Añadido para la lógica de HF

# --- Configuración ---
NOMBRE_DOCUMENTO = "Base de Datos Citas (Proyecto Voz y Chat)"
NOMBRE_HOJA_TRAINING = "Training_NoShow"
MODELO_A_USAR = "logistic" # Puedes cambiar a "knn"
ARCHIVO_MODELO_GUARDADO = "modelo_noshow.joblib"
ARCHIVO_PREPROCESADOR_GUARDADO = "preprocesador_noshow.joblib"

# --- 1. Cargar Datos desde Google Sheets ---
def cargar_datos_gsheets():
    """Carga el dataset de entrenamiento desde Google Sheets."""
    print(f"📥 Cargando datos desde Google Sheets: Hoja '{NOMBRE_HOJA_TRAINING}'...")
    try:
        alcances = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
        
        # --- ⭐️ LÓGICA FUSIONADA (Compatible con HF y Local) ---
        google_creds_json = os.environ.get('GOOGLE_CREDENTIALS_JSON')
        if not google_creds_json:
            print("entrenar_noshow: Secret no encontrado, usando credenciales.json local...")
            cred = Credentials.from_service_account_file("credenciales.json", scopes=alcances)
        else:
            print("entrenar_noshow: Cargando credenciales desde Secret...")
            cred_dict = json.loads(google_creds_json)
            cred = Credentials.from_service_account_info(cred_dict, scopes=alcances)
        
        cliente = gspread.authorize(cred)
        # --- Fin de Lógica Fusionada ---

        documento = cliente.open(NOMBRE_DOCUMENTO)
        hoja = documento.worksheet(NOMBRE_HOJA_TRAINING)
        datos = hoja.get_all_records() # Lee como lista de diccionarios
        if not datos:
            raise ValueError("La hoja de entrenamiento está vacía.")
        print(f"✅ Datos cargados: {len(datos)} filas.")
        return pd.DataFrame(datos)
    except Exception as e:
        print(f"❌ Error al cargar datos: {e}")
        return None

# --- 2. Preparar Datos (Feature Engineering) ---
def preparar_datos(df):
    """Preprocesa los datos para el modelo de ML."""
    print("⚙️ Preparando datos...")
    if df is None or df.empty:
        print("❌ No hay datos para preparar.")
        return None, None, None

    # Verificar columnas necesarias
    columnas_necesarias = ['Dia_Semana', 'Hora_Bloque', 'Ant_No_Shows', 'Distancia_Km', 'asistio']
    faltantes = [col for col in columnas_necesarias if col not in df.columns]
    if faltantes:
        print(f"❌ Faltan columnas en el dataset: {faltantes}")
        return None, None, None

    # a) Definir X (features) e y (target)
    # Seleccionamos las columnas que usará el modelo
    features = ['Dia_Semana', 'Hora_Bloque', 'Ant_No_Shows', 'Distancia_Km']
    X = df[features].copy() # Hacemos una copia para evitar SettingWithCopyWarning
    
    # La columna 'asistio' es nuestro objetivo.
    # Necesitamos convertirla a 0 (No Asistió) y 1 (Sí Asistió).
    # ¡IMPORTANTE! Confirma si en tu hoja 'asistio'=0 significa NO asistió.
    # Si es al revés, invierte la lógica aquí.
    if 'asistio' in df.columns:
         y = df['asistio'].apply(lambda x: 0 if x == 1 else 1) # Asumiendo asistio=1 es SÍ, No_Show=0. Queremos predecir No_Show=1 (faltó)
         print("\nVariable objetivo 'No_Show' (1=Faltó, 0=Asistió):")
         print(y.value_counts())
    else:
        print("❌ Falta la columna objetivo 'asistio'.")
        return None, None, None

    # b) Convertir categóricas a numéricas
    # Usaremos OneHotEncoder para Dia_Semana y Hora_Bloque
    columnas_categoricas = ['Dia_Semana', 'Hora_Bloque']
    columnas_numericas = ['Ant_No_Shows', 'Distancia_Km']

    # Crear el preprocesador
    preprocesador = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), columnas_categoricas), # Convierte Lunes, Martes.. en 0s y 1s
            ('num', 'passthrough', columnas_numericas) # Deja las numéricas como están
        ],
        remainder='drop' # Ignora otras columnas si las hubiera
    )

    # c) Aplicar el preprocesador
    print("Aplicando preprocesamiento (OneHotEncoder)...")
    X_procesado = preprocesador.fit_transform(X)
    print(f"✅ Datos preparados. Shape de X_procesado: {X_procesado.shape}")

    # Guardar el preprocesador para usarlo después en la predicción
    joblib.dump(preprocesador, ARCHIVO_PREPROCESADOR_GUARDADO)
    print(f"💾 Preprocesador guardado en {ARCHIVO_PREPROCESADOR_GUARDADO}")


    return X_procesado, y, preprocesador

# --- 3. Entrenar Modelo ---
def entrenar_modelo(X, y, tipo_modelo="logistic"):
    """Entrena un modelo de clasificación."""
    print(f"\n🧠 Entrenando modelo: {tipo_modelo.upper()}...")

    # Dividir datos en entrenamiento y prueba (80/20)
    # estratificar por 'y' es importante si los datos están desbalanceados
    stratify_target = y if len(y.unique()) > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=stratify_target)
    print(f"Datos divididos: {len(X_train)} para entrenar, {len(X_test)} para probar.")

    if tipo_modelo == "logistic":
        modelo = LogisticRegression(random_state=42, class_weight='balanced') # class_weight ayuda con datos desbalanceados
    elif tipo_modelo == "knn":
        modelo = KNeighborsClassifier(n_neighbors=5) # k=5 es un valor común
    else:
        print(f"❌ Modelo '{tipo_modelo}' no soportado. Usando Regresión Logística.")
        modelo = LogisticRegression(random_state=42, class_weight='balanced')

    modelo.fit(X_train, y_train)
    print("✅ Modelo entrenado.")
    return modelo, X_test, y_test

# --- 4. Evaluar Modelo ---
def evaluar_modelo(modelo, X_test, y_test):
    """Evalúa el rendimiento del modelo entrenado."""
    print("\n📊 Evaluando modelo...")
    y_pred = modelo.predict(X_test)
    y_proba = modelo.predict_proba(X_test)[:, 1] # Probabilidad de la clase '1' (No-Show)

    accuracy = accuracy_score(y_test, y_pred)
    try:
        auc = roc_auc_score(y_test, y_proba)
    except ValueError:
        # AUC no se puede calcular si solo hay una clase en y_test (común con pocos datos)
        auc = float('nan') # Not a Number

    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  AUC: {auc:.4f}")

    print("\n  Matriz de Confusión:")
    # [[Verdaderos Negativos, Falsos Positivos],
    #  [Falsos Negativos,    Verdaderos Positivos]]
    print(confusion_matrix(y_test, y_pred))

    print("\n  Reporte de Clasificación:")
    print(classification_report(y_test, y_pred, zero_division=0))

    # Verificar si cumple el DoD del Sprint 3 (AUC >= 0.75)
    if not pd.isna(auc) and auc >= 0.75:
        print("\n🎉 ¡Meta cumplida! AUC >= 0.75")
    else:
        print("\n⚠️ AUC < 0.75 o no calculable. El modelo necesita mejorar o más datos.")
    return auc # Devolvemos el AUC para decidir si guardar

# --- 5. Guardar Modelo ---
def guardar_modelo(modelo, nombre_archivo):
    """Guarda el modelo entrenado en un archivo."""
    try:
        joblib.dump(modelo, nombre_archivo)
        print(f"\n💾 Modelo entrenado guardado en {nombre_archivo}")
    except Exception as e:
        print(f"❌ Error al guardar el modelo: {e}")


# --- Ejecución Principal ---
if __name__ == "__main__":
    df_entrenamiento = cargar_datos_gsheets()
    if df_entrenamiento is not None:
        X_procesado, y, preprocesador = preparar_datos(df_entrenamiento)

        if X_procesado is not None and y is not None:
             print("\n🚀 ¡Datos listos para entrenar el modelo!")

             # --- Llamadas a las nuevas funciones ---
             modelo_entrenado, X_prueba, y_prueba = entrenar_modelo(X_procesado, y, tipo_modelo=MODELO_A_USAR)
             auc_resultado = evaluar_modelo(modelo_entrenado, X_prueba, y_prueba)

             # Guardamos el modelo solo si la evaluación fue razonable (AUC calculable)
             if not pd.isna(auc_resultado):
                 guardar_modelo(modelo_entrenado, ARCHIVO_MODELO_GUARDADO)
             # --- Fin de las nuevas llamadas ---

        else:
            print("\n❌ No se pudo preparar los datos para el entrenamiento.")
