import pandas as pd
from datetime import date
import spacy
import re
import numpy as np # Necesario para la función predecir_noshow

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
    from procesador_nlp import procesar_texto
    nlp_cargado = True
except ImportError as e:
    print(f"❌ ERROR FATAL: Falló la importación de 'procesador_nlp.py'. Detalle: {e}")
    nlp_cargado = False
    def procesar_texto(texto): return "desconocido", {"error": "Procesador NLP no encontrado."}

# --- Importaciones de Modelo ML (CORRECCIÓN DE NOMBRE DE ARCHIVO) ---
try:
    import joblib
    # 🚨 CORRECCIÓN: Buscamos los archivos .joblib que existen en tu directorio
    modelo_noshow = joblib.load("modelo_noshow.joblib") 
    preprocesador_noshow = joblib.load("preprocesador_noshow.joblib")
    print("✅ chatbot_logic: Modelo ML 'No-Show' cargado.")
    ml_cargado = True
except FileNotFoundError:
    print("❌ ADVERTENCIA: Archivo de modelo ML no encontrado (modelo_noshow.joblib).")
    modelo_noshow, preprocesador_noshow, ml_cargado = None, None, False
except Exception as e:
    print(f"❌ Error chatbot_logic al cargar modelo ML: {e}")
    modelo_noshow, preprocesador_noshow, ml_cargado = None, None, False


# --- Lógica de Predicción No-Show ---
def predecir_noshow(fecha_str, hora_str):
    """Prepara datos y predice la probabilidad de No-Show."""
    if not ml_cargado: return None
    try:
        fecha_obj = pd.to_datetime(fecha_str); dia_semana = fecha_obj.strftime('%A')
        hora_num = int(hora_str.split(':')[0])
        if 5 <= hora_num < 12: hora_bloque = "Mañana"
        elif 12 <= hora_num < 18: hora_bloque = "Tarde"
        else: hora_bloque = "Noche"
        ant_no_shows = 0; distancia_km = 5 # Placeholders
        datos_cita = pd.DataFrame([{'Dia_Semana': dia_semana, 'Hora_Bloque': hora_bloque,'Ant_No_Shows': ant_no_shows, 'Distancia_Km': distancia_km}])
        datos_procesados = preprocesador_noshow.transform(datos_cita)
        
        # El modelo espera un array 2D de features (ajusta según tu modelo real)
        # Nota: Asegúrate de que preprocesador_noshow esté cargado correctamente
        prob = modelo_noshow.predict_proba(datos_procesados)[0][1]
        
        print(f"📈 chatbot_logic: Predicción No-Show ({fecha_str} {hora_str}): {prob:.2f}"); return prob
    except Exception as e: print(f"❌ chatbot_logic: Error en predicción: {e}"); return None


# --- Función Principal del Chatbot (Estado) ---
def responder_chatbot(mensaje, historial_chat, estado_actual):
    """
    Función principal del chatbot con flujo conversacional para agendar.
    """
    respuesta = ""
    accion_completada = False
    
    # 🚨 CORRECCIÓN DE SEGURIDAD: Aseguramos que el estado inicial sea un diccionario
    if estado_actual is None: estado_actual = {}
    print(f"Estado IN: {estado_actual}")

    campos_paciente = ["DNI", "Nombre", "Telefono", "Email"]
    campos_cita = ["Fecha", "Hora", "Medico"]
    todos_campos = campos_paciente + campos_cita

    if not nlp_cargado: 
        # Si NLP no cargó, devolvemos el error de texto del fallback
        return "Error: El módulo NLP no está disponible.", estado_actual

    # Si el mensaje es un estado de error, lo limpiamos y devolvemos un mensaje de inicio.
    if isinstance(mensaje, str) and mensaje.startswith("Error:"):
         respuesta = "Hubo un error de formato. Por favor, reinicia la conversación."
         return respuesta, {}


    intencion_raw, entidades_raw = procesar_texto(mensaje)
    print(f"Intención RAW: {intencion_raw}, Entidades RAW: {entidades_raw}")
    
    # 2. Lógica de Reinicio o Cambio de Intención
    if intencion_raw in ["saludo", "desconocido"]:
        respuesta = "Hola. Puedo ayudarte a agendar, consultar o cancelar citas."
        return respuesta, {} 

    if estado_actual.get("intent") and estado_actual["intent"] != intencion_raw and intencion_raw not in ["saludo", "desconocido"]:
        estado_actual = {} 
        estado_actual["intent"] = intencion_raw
        respuesta = f"Entendido, vamos a empezar de nuevo con la acción de '{intencion_raw}'."
    
    if not estado_actual.get("intent"):
        estado_actual["intent"] = intencion_raw

    # 3. Limpiar y Consolidar Entidades (Lógica de consolidación)
    entidades_limpias = {k: v for k, v in entidades_raw.items() if v}
    estado_actual.update(entidades_limpias)
    
    # ... (Se omite el resto de la lógica de flujo conversacional para brevedad, asumiendo que es idéntica)

    # 4. Lógica de Flujo (Estado y Respuesta)
    if estado_actual.get("intent") == "agendar":
        if not flujo_cargado: return "Error: La lógica de agendamiento no está disponible.", {}
        
        campos_pendientes = [c for c in CAMPOS_AGENDAR if c not in estado_actual]
        
        if not campos_pendientes:
            # Todos los campos listos
            try:
                # 1. Buscar si el paciente existe y consolidar datos
                paciente = buscar_paciente_por_dni(estado_actual["DNI"])
                if paciente is None:
                    nombre, telefono, email = estado_actual["Nombre"], estado_actual["Telefono"], estado_actual["Email"]
                else:
                    nombre = paciente.get("Nombre", estado_actual["Nombre"]) 
                    telefono = paciente.get("Telefono", estado_actual["Telefono"])
                    email = paciente.get("Email", estado_actual["Email"])

                # 2. Agendar 
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
        respuesta = "Disculpa, tengo un problema interno. Por favor, reinicia el chat."
        estado_actual = {}

    
    # 5. Devolver Respuesta y Estado
    
    # 🚨 VALIDACIÓN DE SEGURIDAD (Se mantiene la validación anterior)
    if not isinstance(respuesta, str):
        print("⚠️ Alerta: La respuesta final no es una cadena. Forzando a string.")
        # Esto previene el error de validación de Gradio/Pydantic V2
        respuesta = "Error interno de formato (DEBUG). Por favor, reinicia la conversación."

    # El retorno siempre debe ser una tupla (string, dict) para Gradio
    return respuesta, estado_actual
