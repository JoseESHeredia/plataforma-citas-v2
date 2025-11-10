import pandas as pd
from datetime import date
import re
import numpy as np 
import unicodedata # ⭐️ Añadido para normalizar nombres
import dateparser # ⭐️ Añadido para validar fechas

# =========================================================
# 🚨 CONSTANTES DE LÓGICA
# =========================================================

CAMPOS_AGENDAR = ["DNI", "Nombre", "Telefono", "Email", "Medico", "Fecha", "Hora"]
CAMPOS_PACIENTE = ["DNI", "Nombre", "Telefono", "Email"]
RESPUESTAS_PREGUNTAS = {
    "DNI": "¿Cuál es tu número de DNI?",
    "Nombre": "¿Cuál es tu nombre completo?",
    "Telefono": "¿Me proporcionas un número de teléfono de 9 dígitos que empiece con 9?",
    "Email": "¿Me das tu email?",
    "Medico": "Pregunta de Médico (será reemplazada)", # Se genera dinámicamente
    "Fecha": "¿Qué fecha quieres la cita?",
    "Hora": "¿A qué hora? (Ej. 3pm o 15:00)"
}

# =========================================================
# 🔧 IMPORTACIONES DE LÓGICA EXTERNA
# =========================================================

try:
    from flujo_agendamiento import (
        agendar, 
        consultar_citas, 
        cancelar_cita, 
        obtener_medicos, 
        buscar_paciente_por_dni,
        asignar_especialidad
    )
    flujo_cargado = True
    
    # ⭐️ Creamos la lista formateada de médicos y sus especialidades (Bug C)
    LISTA_MEDICOS_TEXTO = "\nNuestros especialistas disponibles son:\n"
    MEDICOS_VALIDOS = obtener_medicos()
    for med in MEDICOS_VALIDOS:
        LISTA_MEDICOS_TEXTO += f"* {med} ({asignar_especialidad(med)})\n"
    
    # ⭐️ Actualizamos la pregunta de Médico
    RESPUESTAS_PREGUNTAS["Medico"] = f"¿Con qué especialista deseas agendar? {LISTA_MEDICOS_TEXTO}"

except ImportError as e:
    print(f"❌ ERROR chatbot_logic: No se encontró 'flujo_agendamiento.py': {e}")
    flujo_cargado = False
    def agendar(*args): return "Error: Lógica de agendamiento no encontrada."
    def consultar_citas(dni): return "Error: Lógica de consulta no encontrada."
    def cancelar_cita(dni, fecha): return "Error: Lógica de cancelación no encontrada."
    def obtener_medicos(): return ["Error"]
    def buscar_paciente_por_dni(dni): return None
    LISTA_MEDICOS_TEXTO = "Error al cargar médicos."
    MEDICOS_VALIDOS = []


try:
    from procesador_nlp import procesar_texto
    nlp_cargado = True
except ImportError as e:
    print(f"❌ ERROR FATAL: Falló la importación de 'procesador_nlp.py'. Detalle: {e}")
    nlp_cargado = False
    def procesar_texto(texto): return "desconocido", {"error": "Procesador NLP no encontrado."}

# --- Importaciones de Modelo ML ---
try:
    import joblib
    modelo_noshow = joblib.load("modelo_noshow.joblib") 
    preprocesador_noshow = joblib.load("preprocesador_noshow.joblib")
    print("✅ chatbot_logic: Modelo ML 'No-Show' cargado.")
    ml_cargado = True
except FileNotFoundError:
    modelo_noshow, preprocesador_noshow, ml_cargado = None, None, False
except Exception as e:
    print(f"❌ Error chatbot_logic al cargar modelo ML: {e}")
    modelo_noshow, preprocesador_noshow, ml_cargado = None, None, False


# =========================================================
# 🤖 LÓGICA AUXILIAR DEL BOT
# =========================================================

def predecir_noshow(fecha_str, hora_str):
    if not ml_cargado: return None
    try:
        fecha_obj = pd.to_datetime(fecha_str); dia_semana = fecha_obj.strftime('%A')
        hora_num = int(hora_str.split(':')[0])
        if 5 <= hora_num < 12: hora_bloque = "Mañana"
        elif 12 <= hora_num < 18: hora_bloque = "Tarde"
        else: hora_bloque = "Noche"
        # Usamos valores placeholder para la predicción
        ant_no_shows = 0; distancia_km = 5 
        datos_cita = pd.DataFrame([{'Dia_Semana': dia_semana, 'Hora_Bloque': hora_bloque,'Ant_No_Shows': ant_no_shows, 'Distancia_Km': distancia_km}])
        datos_procesados = preprocesador_noshow.transform(datos_cita)
        prob = modelo_noshow.predict_proba(datos_procesados)[0][1]
        print(f"📈 chatbot_logic: Predicción No-Show ({fecha_str} {hora_str}): {prob:.2f}"); return prob
    except Exception as e: print(f"❌ chatbot_logic: Error en predicción: {e}"); return None

# ⭐️ NUEVA FUNCIÓN: Normalizar nombres de médicos (Bug C)
def normalizar_texto(texto):
    if not isinstance(texto, str): return ""
    texto = texto.lower()
    texto = re.sub(r'\b(dr|dra|doctor|doctora)\b\.?', '', texto).strip()
    texto = ''.join(c for c in unicodedata.normalize('NFD', texto) if unicodedata.category(c) != 'Mn')
    texto = re.sub(r'[\s\.]+', '', texto)
    return texto

def encontrar_medico(texto_usuario, medicos_validos):
    texto_norm = normalizar_texto(texto_usuario)
    if not texto_norm: return None
    # Corregido: Normalizar las keys del mapeo antes de buscar.
    medicos_norm_map = {normalizar_texto(m): m for m in medicos_validos} 
    
    for med_norm, med_original in medicos_norm_map.items():
        if med_norm in texto_norm:
            return med_original 
    return None

# ⭐️ NUEVA FUNCIÓN: Validar formato (Bug E + Bug 2)
def validar_formato(campo, valor):
    if campo == "DNI":
        dni_limpio = ''.join(filter(str.isdigit, str(valor)))
        if len(dni_limpio) == 8:
            return dni_limpio, None # Valor limpio, Sin error
        return valor, "El DNI debe tener 8 dígitos."
    
    if campo == "Telefono":
        tel_limpio = ''.join(filter(str.isdigit, str(valor)))
        if len(tel_limpio) == 9 and tel_limpio.startswith("9"):
            return tel_limpio, None
        return valor, "El teléfono debe tener 9 dígitos y empezar con 9."
    
    if campo == "Fecha":
        # Usamos dateparser para validar (ej. rechazar "15811/2025")
        fecha_obj = dateparser.parse(valor, languages=['es'])
        # FIX: Ahora compara con la fecha de hoy, no con el tiempo exacto.
        if fecha_obj and fecha_obj.date() >= date.today(): 
            return fecha_obj.strftime("%Y-%m-%d"), None # Formatear a AAAA-MM-DD
        return valor, "No entendí esa fecha o es una fecha pasada. Por favor, dime una fecha futura y válida (ej. 'mañana', '2026-01-15')."

    # El resto (Nombre, Email, Hora) se acepta tal cual (el NLP ya los formateó).
    return str(valor), None


# =========================================================
# 🧠 FUNCIÓN PRINCIPAL DEL CHATBOT (CON ESTADO)
# =========================================================
def responder_chatbot(mensaje, historial_chat, estado_actual):
    """
    Función principal del chatbot con flujo conversacional mejorado.
    """
    respuesta = ""
    if estado_actual is None: estado_actual = {}
    print(f"\n--- Turno Nuevo ---")
    print(f"Estado IN: {estado_actual}")
    print(f"Mensaje IN: {mensaje}")

    if not nlp_cargado or not flujo_cargado: 
        return "Error: Los módulos de NLP o Flujo no están disponibles.", {}

    # 1. Procesar respuesta a confirmación (Bug F)
    if estado_actual.get("confirmando_agendar"):
        del estado_actual["confirmando_agendar"]
        if "si" in mensaje.lower() or "sí" in mensaje.lower():
            print("Confirmación recibida. Agendando...")
            try:
                # Los datos en estado_actual ya están limpios y validados
                res_agendar = agendar(
                    estado_actual["Nombre"], estado_actual["DNI"], estado_actual["Telefono"], 
                    estado_actual["Email"], estado_actual["Fecha"], estado_actual["Hora"], estado_actual["Medico"]
                )
                prob = predecir_noshow(estado_actual["Fecha"], estado_actual["Hora"])
                respuesta = res_agendar
                if prob is not None:
                     respuesta += f"\n{'⚠️ Riesgo ausencia:' if prob>0.6 else '(Riesgo bajo:'} {prob:.0%})"
                estado_actual = {} # Éxito, limpiar estado
                return respuesta, estado_actual
            except Exception as e:
                respuesta = f"Error al agendar: {e}."
                return respuesta, {}
        else:
            # Resetear estado al decir "no"
            print("Confirmación rechazada. Reseteando estado.")
            respuesta = "OK, se cancela el agendamiento. ¿En qué te puedo ayudar ahora?"
            estado_actual = {} 
            return respuesta, estado_actual

    # 2. Obtener NLP y gestionar el estado
    campo_pendiente = estado_actual.get("campo_preguntado")
    intencion_actual = estado_actual.get("intent")
    intencion_raw, entidades_raw = procesar_texto(mensaje)
    print(f"NLP RAW: Intención={intencion_raw}, Entidades={entidades_raw}")

    INTENCIONES_PRINCIPALES = ["agendar", "consultar", "cancelar"]

    # =========================================================
    # ⭐️ LÓGICA DE PRIORIDADES ⭐️
    # =========================================================

    # PRIORIDAD 1: ¿Quiere el usuario cambiar de tema?
    if intencion_raw in INTENCIONES_PRINCIPALES and intencion_raw != intencion_actual:
        print(f"FIX (P1): CAMBIO DE INTENCIÓN CLARO. De '{intencion_actual}' a '{intencion_raw}'. Reiniciando.")
        
        entidades_limpias = {k: v for k, v in entidades_raw.items() if v}
        
        estado_actual = {} # Reset total
        
        # Aplicar la nueva intención y sus entidades
        estado_actual["intent"] = intencion_raw
        estado_actual.update(entidades_limpias)

        # Actualizar variables locales para el flujo de este turno
        intencion_actual = intencion_raw
        campo_pendiente = None 

    # PRIORIDAD 2: Si no... ¿Me está respondiendo?
    elif campo_pendiente:
        print(f"FIX (P2): 'Sticky Intent'. El usuario está respondiendo. Manteniendo '{intencion_actual}'.")
        
        # Forzar la intención actual (la del flujo que estamos siguiendo)
        intencion_raw = intencion_actual 
        
        # Si el NLP no extrajo la entidad, coger el texto entero del mensaje
        if campo_pendiente not in entidades_raw:
            entidades_raw[campo_pendiente] = mensaje.strip()
        
        # FIX CRÍTICO: Limpiar el flag de "pregunta pendiente" para que el flujo avance
        if "campo_preguntado" in estado_actual:
            del estado_actual["campo_preguntado"]

    # PRIORIDAD 3: Es el primer turno o una continuación sin estado
    elif not intencion_actual:
        estado_actual["intent"] = intencion_raw
        intencion_actual = intencion_raw 
        
    # =========================================================
    # ⭐️ FIN DE LA LÓGICA DE PRIORIDADES ⭐️
    # =========================================================

    # 4. Consolidar Entidades (Solo si no están validadas)
    entidades_limpias = {k: v for k, v in entidades_raw.items() if v}
    for k, v in entidades_limpias.items():
        if not estado_actual.get(f"{k}_validado"): # No sobrescribir datos ya validados
            estado_actual[k] = v
    
    # 5. Lógica de Flujo por Intención
    
    # =========================================================
    # ➡️ FLUJO: AGENDAR (SOLUCIÓN AL LOOP INFINITO)
    # =========================================================
    if estado_actual.get("intent") == "agendar":
        print("Flujo AGENDAR.")
        
        # ⭐️ INICIO DE LÓGICA DE AVANCE ÚNICO Y SEGURO ⭐️
        
        # --- 1. Validar/Buscar DNI (Solo una vez) ---
        if "DNI" in estado_actual and not estado_actual.get("DNI_validado"):
            dni_valor = estado_actual["DNI"]
            valor_limpio, error_formato = validar_formato("DNI", dni_valor)
            
            if error_formato:
                respuesta = f"{error_formato} {RESPUESTAS_PREGUNTAS['DNI']}"
                del estado_actual["DNI"]
                estado_actual["campo_preguntado"] = "DNI"
                return respuesta, estado_actual
            
            # DNI limpio. Buscar paciente
            estado_actual["DNI"] = valor_limpio
            paciente = buscar_paciente_por_dni(valor_limpio)
            
            if paciente:
                print(f"FIX (Bug B): Paciente encontrado: {paciente['Nombre']}.")
                estado_actual.update(paciente) 
                # Marcar DNI y otros campos de paciente como VALIDADOS para que el bucle los salte
                estado_actual["DNI_validado"] = True
                estado_actual["Nombre_validado"] = True
                estado_actual["Telefono_validado"] = True
                estado_actual["Email_validado"] = True
                respuesta = f"¡Hola de nuevo, {paciente['Nombre']}! Ya tengo tus datos. "
                # El siguiente campo a pedir es Medico
                siguiente_campo = "Medico"
            else:
                print(f"FIX (Bug B): Paciente {valor_limpio} no encontrado. Es nuevo.")
                estado_actual["DNI_validado"] = True # Marcar DNI como válido.
                respuesta = "Eres un paciente nuevo. Necesitaré unos datos más. "
                # El siguiente campo a pedir es Nombre
                siguiente_campo = "Nombre"
            
            # Devolver respuesta y setear el siguiente campo
            respuesta += RESPUESTAS_PREGUNTAS[siguiente_campo]
            estado_actual["campo_preguntado"] = siguiente_campo
            return respuesta, estado_actual
        
        # --- 2. Bucle para el resto de los campos (Avance secuencial) ---
        for campo in CAMPOS_AGENDAR:
            
            # Si el campo ya está validado (incluyendo DNI/datos de paciente existente), continuar
            if estado_actual.get(f"{campo}_validado"):
                continue

            # 3. ¿Falta el campo? (Solo debe faltar si no fue validado en este turno)
            if campo not in estado_actual:
                # Esto sucede si no se proveyó la entidad en el turno anterior.
                respuesta = RESPUESTAS_PREGUNTAS[campo]
                estado_actual["campo_preguntado"] = campo
                return respuesta, estado_actual
            
            # 4. El campo existe pero no está validado. Validar AHORA.
            valor = estado_actual[campo]
            
            # --- Validar Formato (Bug E + Bug 4) ---
            valor_limpio, error_formato = validar_formato(campo, valor)
            if error_formato:
                print(f"FIX (Bug E/4): Error de formato en {campo} ('{valor}').")
                respuesta = f"{error_formato} {RESPUESTAS_PREGUNTAS[campo]}"
                del estado_actual[campo] # Borra el dato inválido
                estado_actual["campo_preguntado"] = campo
                return respuesta, estado_actual
            estado_actual[campo] = valor_limpio # Guardar valor limpio/formateado

            # --- Validar Médico (Bug C) ---
            if campo == "Medico":
                medico_encontrado = encontrar_medico(valor_limpio, MEDICOS_VALIDOS)
                if not medico_encontrado:
                    print(f"FIX (Bug C): Médico '{valor_limpio}' NO es válido.")
                    respuesta = f"Lo siento, no encontré un médico llamado '{valor_limpio}'. {LISTA_MEDICOS_TEXTO} ¿Con cuál de ellos deseas agendar?"
                    del estado_actual["Medico"] 
                    estado_actual["campo_preguntado"] = "Medico" 
                    return respuesta, estado_actual
                else:
                    estado_actual["Medico"] = medico_encontrado
            
            estado_actual[f"{campo}_validado"] = True # Marcar como listo

            # Si el campo fue validado con éxito, preguntamos por el siguiente
            try:
                indice_actual = CAMPOS_AGENDAR.index(campo)
                siguiente_campo = CAMPOS_AGENDAR[indice_actual + 1]
                
                # Si el siguiente campo ya está validado (caso DNI -> Nombre, si DNI existe), saltamos este paso.
                if not estado_actual.get(f"{siguiente_campo}_validado"):
                    respuesta = RESPUESTAS_PREGUNTAS[siguiente_campo]
                    estado_actual["campo_preguntado"] = siguiente_campo
                    return respuesta, estado_actual
            except IndexError:
                pass # Es el último campo (Hora)

        # Si el bucle 'for' termina, significa que los 7 campos están presentes Y validados
        
        # --- Resumen de Confirmación (Bug F) ---
        print("FIX (Bug F): Todos los campos listos. Mostrando resumen.")
        # La búsqueda se realiza en flujo_agendamiento, pero para el resumen usamos la versión local.
        paciente_tipo = "Existente" if buscar_paciente_por_dni(estado_actual["DNI"]) else "Nuevo"
        especialidad = asignar_especialidad(estado_actual["Medico"])
        
        respuesta = (
            f"¡Perfecto! Por favor, confirma tus datos:\n\n"
            f"**Paciente:** {estado_actual['Nombre']} ({paciente_tipo})\n"
            f"**DNI:** {estado_actual['DNI']}\n"
            f"**Teléfono:** {estado_actual['Telefono']}\n"
            f"**Médico:** {estado_actual['Medico']} ({especialidad})\n"
            f"**Cita:** {estado_actual['Fecha']} a las {estado_actual['Hora']}\n\n"
            "¿Es todo correcto? (Responde 'Sí' o 'No')"
        )
        estado_actual["confirmando_agendar"] = True
        return respuesta, estado_actual


    # =========================================================
    # ➡️ FLUJO: CANCELAR (Corregido limpieza de estado)
    # =========================================================
    elif estado_actual.get("intent") == "cancelar":
        print("Flujo CANCELAR.")

        # --- Flujo de confirmación de ID de cita (Bug H) ---
        if estado_actual.get("campo_preguntado") == "cancelar_id":
            citas_pendientes = estado_actual.get("citas_pendientes", [])
            dni_a_usar = estado_actual.get("DNI")

            cita_a_cancelar = None
            for c in citas_pendientes:
                if mensaje.strip().lower() in c['ID_Cita'].lower() or mensaje.strip() in c['Fecha']:
                    cita_a_cancelar = c
                    break
            
            if cita_a_cancelar:
                respuesta = cancelar_cita(dni_a_usar, cita_a_cancelar['Fecha'])
                estado_actual = {} # FIX: Resetear estado al finalizar
            else:
                respuesta = "No entendí esa selección. Por favor, dime la **fecha exacta (AAAA-MM-DD)** o el **ID de la cita (ej. C022)** de la lista de arriba."
                estado_actual["campo_preguntado"] = "cancelar_id" # Volver a preguntar
            return respuesta, estado_actual

        # --- Validar DNI (Bug E) ---
        if "DNI" in estado_actual and not estado_actual.get("DNI_validado"):
            valor, error_formato = validar_formato("DNI", estado_actual["DNI"])
            if error_formato:
                respuesta = f"{error_formato} {RESPUESTAS_PREGUNTAS['DNI']}"
                del estado_actual["DNI"]
                estado_actual["campo_preguntado"] = "DNI"
                return respuesta, estado_actual
            estado_actual["DNI"] = valor
            estado_actual["DNI_validado"] = True
        
        # --- Pedir DNI si falta ---
        if not estado_actual.get("DNI_validado"):
            respuesta = "Necesito tu DNI para cancelar."
            estado_actual["campo_preguntado"] = "DNI"
        
        # --- Mostrar lista de citas (Bug H) ---
        else:
            print("FIX (Bug H): DNI válido. Buscando citas pendientes...")
            res_crud = consultar_citas(estado_actual["DNI"])
            
            if not isinstance(res_crud, list):
                estado_actual = {} # FIX: Resetear estado
                return res_crud, estado_actual 
            
            pendientes = [c for c in res_crud if c.get('Estado').lower() == "pendiente"]
            
            if not pendientes:
                respuesta = f"No encontré citas 'Pendientes' para el DNI {estado_actual['DNI']}."
                estado_actual = {} # FIX: Resetear estado
            else:
                respuesta = f"He encontrado {len(pendientes)} cita(s) pendiente(s) para DNI {estado_actual['DNI']}:\n"
                for c in pendientes:
                    respuesta += f"* Cita **{c.get('ID_Cita','N/A')}** el **{c.get('Fecha','N/A')}** a las {c.get('Hora','N/A')} (con {c.get('Medico','N/A')})\n"
                respuesta += "\n¿Cuál de estas deseas cancelar? (Dime la fecha o el ID de la cita)"
                
                estado_actual["citas_pendientes"] = pendientes
                estado_actual["campo_preguntado"] = "cancelar_id" # Pedir la confirmación/ID
    
        return respuesta, estado_actual 

    # =========================================================
    # ➡️ FLUJO: CONSULTAR (Corregido limpieza de estado)
    # =========================================================
    elif estado_actual.get("intent") == "consultar":
        print("Flujo CONSULTAR.")
        
        # --- Validar DNI (Bug E) ---
        if "DNI" in estado_actual and not estado_actual.get("DNI_validado"):
            valor, error_formato = validar_formato("DNI", estado_actual["DNI"])
            if error_formato:
                respuesta = f"{error_formato} {RESPUESTAS_PREGUNTAS['DNI']}"
                del estado_actual["DNI"]
                estado_actual["campo_preguntado"] = "DNI"
                return respuesta, estado_actual
            estado_actual["DNI"] = valor
            estado_actual["DNI_validado"] = True
        
        if not estado_actual.get("DNI_validado"):
            respuesta = "Necesito tu DNI para consultar."
            estado_actual["campo_preguntado"] = "DNI"
        else:
            res_crud = consultar_citas(estado_actual["DNI"])
            if isinstance(res_crud, list):
                if not res_crud: 
                    respuesta = f"No encontré citas para DNI {estado_actual['DNI']}."
                else:
                    respuesta = f"He encontrado {len(res_crud)} citas (incluyendo historial) para DNI {estado_actual['DNI']}:\n"
                    for c in res_crud: 
                        respuesta += f"* Cita **{c.get('ID_Cita','N/A')}** el **{c.get('Fecha','N/A')}** a las {c.get('Hora','N/A')} (Estado: {c.get('Estado','N/A')})\n"
            else: 
                respuesta = str(res_crud) # Error "No se encontró paciente..."
            estado_actual = {} # FIX: Resetear estado al finalizar

    elif estado_actual.get("intent") == "desconocido":
        print("Flujo DESCONOCIDO.")
        respuesta = "No entendí. Intenta: agendar, consultar o cancelar."
        estado_actual = {} # FIX: Resetear estado

    elif not respuesta:
        print("Flujo ERROR INTERNO.")
        respuesta = "Disculpa, tengo un problema interno o necesito más información. Por favor, reinicia el chat diciendo qué quieres hacer (agendar, consultar o cancelar)."
        estado_actual = {} # FIX: Resetear estado

    
    # 6. Devolver Respuesta y Estado
    print(f"Estado OUT: {estado_actual}")
    print(f"Respuesta OUT: {respuesta}")
    return respuesta, estado_actual
