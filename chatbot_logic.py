import pandas as pd
from datetime import date
import spacy # <-- AÑADIDO
import re # <-- AÑADIDO

# --- Importaciones de Lógica Externa ---
try:
    from flujo_agendamiento import agendar, consultar_citas, cancelar_cita, obtener_medicos, buscar_paciente_por_dni
    flujo_cargado = True
except ImportError:
    print("ERROR chatbot_logic: No se encontró 'flujo_agendamiento.py'")
    flujo_cargado = False
    # Define placeholders
    def agendar(*args): return "Error: Lógica de agendamiento no encontrada."
    def consultar_citas(dni): return "Error: Lógica de consulta no encontrada."
    def cancelar_cita(dni, fecha): return "Error: Lógica de cancelación no encontrada."
    def obtener_medicos(): return ["Error"]
    def buscar_paciente_por_dni(dni): return None

try:
    from procesador_nlp import procesar_texto # <-- ESTA LÍNEA DEBE SER CORRECTA
    nlp_cargado = True
except ImportError as e:
    print(f"ERROR chatbot_logic: No se encontró 'procesador_nlp.py'. Detalle: {e}") # <-- Mensaje de error mejorado
    nlp_cargado = False
    def procesar_texto(texto): return "desconocido", {"error": "Procesador NLP no encontrado."}

# --- Importaciones de Modelo ML ---
try:
    import joblib
    import numpy as np
    modelo_noshow = joblib.load('modelo/modelo_noshow.pkl')
    print("✅ chatbot_logic: Modelo ML 'No-Show' cargado.")
except FileNotFoundError:
    print("ADVERTENCIA: Archivo 'modelo_noshow.pkl' no encontrado.")
    modelo_noshow = None
except Exception as e:
    print(f"ADVERTENCIA: Error al cargar modelo ML: {e}")
    modelo_noshow = None

# --- Predicción No-Show (Simple) ---
def predecir_noshow(fecha_str, hora_str):
    """
    Genera una probabilidad de no-show (ejemplo simple).
    Devuelve None si el modelo no está cargado.
    """
    if modelo_noshow is None: return None
    
    # Lógica de ejemplo: Si es fin de mes y tarde, más riesgo
    try:
        fecha = date.fromisoformat(fecha_str)
        hora = int(hora_str.split(':')[0]) # Obtener la hora como entero
        
        # Simulación de características (ejemplo simple para el modelo)
        # 1. Dia de la semana (Lunes=0, Domingo=6)
        dia_semana = fecha.weekday() 
        # 2. Última semana del mes (simplificado: días 25-31)
        es_fin_mes = 1 if fecha.day >= 25 else 0 
        # 3. Tarde (después de las 16:00)
        es_tarde = 1 if hora >= 16 else 0
        
        # El modelo espera un array 2D de features (ajusta según tu modelo real)
        features = np.array([[dia_semana, es_fin_mes, es_tarde]]) 
        
        # Probabilidad de la clase 1 (No-Show)
        prob_noshow = modelo_noshow.predict_proba(features)[:, 1][0] 
        
        return prob_noshow
    except Exception as e:
        print(f"Error en predecir_noshow: {e}")
        return None


# --- Definiciones de Flujo (Simulación de Turnos) ---
CAMPOS_AGENDAR = ["DNI", "Nombre", "Telefono", "Email", "Medico", "Fecha", "Hora"]
RESPUESTAS_PREGUNTAS = {
    "DNI": "¿Cuál es tu número de DNI?",
    "Nombre": "¿Cuál es tu nombre completo?",
    "Telefono": "¿Me proporcionas un número de teléfono?",
    "Email": "¿Me das tu email?",
    "Medico": "¿Con qué médico quieres agendar? Tenemos al Dr. Vega, Dra. Perez, o Dr. Morales.",
    "Fecha": "¿Qué fecha quieres la cita? (Formato AAAA-MM-DD)",
    "Hora": "¿A qué hora? (Formato HH:MM)"
}


# --- Función Principal del Chatbot (Estado) ---
def responder_chatbot(mensaje, historial_chat, estado_actual):
    """
    Procesa el mensaje, mantiene el estado de la conversación y genera una respuesta.
    """
    respuesta = ""
    accion_completada = False

    # 1. Procesamiento NLP (Intención y Entidades)
    if not nlp_cargado: return "Error: Lógica de NLP no cargada.", {}

    intencion_raw, entidades_raw = procesar_texto(mensaje)
    print(f"Intención RAW: {intencion_raw}, Entidades RAW: {entidades_raw}")
    
    # 2. Lógica de Reinicio o Cambio de Intención
    # Si la intención es 'saludo' o 'desconocido', o si la intención raw difiere del estado, reiniciar.
    if intencion_raw in ["saludo", "desconocido"]:
        respuesta = "Hola. Puedo ayudarte a agendar, consultar o cancelar citas."
        return respuesta, {} # Reiniciar el estado

    if estado_actual.get("intent") and estado_actual["intent"] != intencion_raw and intencion_raw not in ["saludo", "desconocido"]:
        # Si el usuario cambia de tema, forzar un reinicio al nuevo tema
        estado_actual = {} 
        estado_actual["intent"] = intencion_raw
        respuesta = f"Entendido, vamos a empezar de nuevo con la acción de '{intencion_raw}'."
    
    if not estado_actual.get("intent"):
        estado_actual["intent"] = intencion_raw

    # 3. Limpiar y Consolidar Entidades
    entidades_limpias = {k: v for k, v in entidades_raw.items() if v}
    
    # Mover entidades al estado si la intención es la misma
    estado_actual.update(entidades_limpias)
    
    # 4. Lógica de Flujo (Estado y Respuesta)
    if estado_actual.get("intent") == "agendar":
        if not flujo_cargado: return "Error: Lógica de agendamiento no disponible.", {}
        
        campos_pendientes = [c for c in CAMPOS_AGENDAR if c not in estado_actual]
        
        if not campos_pendientes:
            # Todos los campos están listos, proceder a agendar
            try:
                # 1. Buscar si el paciente existe
                paciente = buscar_paciente_por_dni(estado_actual["DNI"])
                if paciente is None:
                    # Si no existe, usamos los datos del chat
                    nombre = estado_actual["Nombre"]
                    telefono = estado_actual["Telefono"]
                    email = estado_actual["Email"]
                else:
                    # Si existe, sobrescribimos con los datos de GSheets
                    nombre = paciente.get("Nombre", estado_actual["Nombre"]) 
                    telefono = paciente.get("Telefono", estado_actual["Telefono"])
                    email = paciente.get("Email", estado_actual["Email"])

                # 2. Agendar (el DNI lo tenemos en el estado)
                res_agendar = agendar(nombre, estado_actual["DNI"], telefono, email, estado_actual["Fecha"], estado_actual["Hora"], estado_actual["Medico"])

                # 3. Predecir No-Show
                prob = predecir_noshow(estado_actual["Fecha"], estado_actual["Hora"])

                respuesta = res_agendar
                if prob is not None:
                     respuesta += f"\n{'⚠️ Riesgo ausencia:' if prob>0.6 else '(Riesgo bajo:'} {prob:.0%})"
                
                estado_actual = {} # Limpiar estado
                accion_completada = True

            except Exception as e:
                respuesta = f"Error al agendar: {e}. Por favor, revisa tus datos."
                # No limpiar el estado para que pueda intentarlo de nuevo
                accion_completada = True 

        else:
            # Pedir el siguiente campo pendiente
            campo_a_pedir = campos_pendientes[0]
            respuesta = RESPUESTAS_PREGUNTAS[campo_a_pedir]
            estado_actual["campo_preguntado"] = campo_a_pedir


    elif estado_actual.get("intent") == "consultar":
        if not flujo_cargado: return "Error: Lógica de consulta no disponible.", {}
        dni = estado_actual.get("DNI") or entidades_limpias.get("DNI")
        if not dni: 
            respuesta = "Necesito tu DNI para consultar."
            estado_actual["campo_preguntado"] = "DNI"
        else:
            res_crud = consultar_citas(dni)
            if isinstance(res_crud, list):
                if not res_crud: respuesta = f"No encontré citas para DNI {dni}."
                else:
                    respuesta = f"He encontrado {len(res_crud)} citas para DNI {dni}:\n"
                    for c in res_crud: respuesta += f"- {c.get('ID_Cita','N/A')} el {c.get('Fecha','N/A')} {c.get('Hora','N/A')} ({c.get('Estado','N/A')})\n"
            else: respuesta = str(res_crud)
            estado_actual = {} # Limpiar estado
        accion_completada = True

    elif estado_actual.get("intent") == "cancelar":
        if not flujo_cargado: return "Error: Lógica de cancelación no disponible.", {}
        dni = estado_actual.get("DNI") or entidades_limpias.get("DNI")
        fecha = estado_actual.get("Fecha") or entidades_limpias.get("Fecha")
        
        if not dni:
            respuesta = "Necesito tu DNI para cancelar."
            estado_actual["campo_preguntado"] = "DNI"
        elif not fecha:
            respuesta = "¿Para qué fecha es la cita que quieres cancelar? (AAAA-MM-DD)"
            estado_actual["campo_preguntado"] = "Fecha"
        else: 
            respuesta = cancelar_cita(dni, fecha)
            estado_actual = {} # Limpiar estado
        accion_completada = True

    elif estado_actual.get("intent") == "desconocido":
        respuesta = "No entendí. Intenta: agendar, consultar o cancelar."
        estado_actual = {} # Limpiar estado
        accion_completada = True

    elif not respuesta:
        # Esto solo debería ocurrir si hay un fallo en la lógica de flujo
        respuesta = "Disculpa, tengo un problema interno. Por favor, reinicia el chat."
        estado_actual = {}


    # 5. Devolver Respuesta y Estado
    
    # 🚨 VALIDACIÓN DE SEGURIDAD CONTRA EL ERROR DE GRADIO/PYDANTIC V2
    # Aseguramos que el primer elemento devuelto (la respuesta del chat) sea SIEMPRE un string.
    if not isinstance(respuesta, str):
        print("⚠️ Alerta: La respuesta final no es una cadena. Forzando a string.")
        # Intentamos usar una respuesta coherente, sino un mensaje de error
        if isinstance(respuesta, dict) and respuesta.get('intent'):
             respuesta = f"Entendido, vamos a proceder con: {respuesta.get('intent')}. ¿Cuál es el siguiente dato?"
        else:
             respuesta = "Error de formato interno. Por favor, reinicia la conversación."

    # El retorno siempre debe ser una tupla (string, dict) para Gradio
    return respuesta, estado_actual
