# chatbot_logic.py
import pandas as pd # Necesario para predecir_noshow
from datetime import date # Necesario para predecir_noshow

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
except ImportError:
    print("ERROR chatbot_logic: No se encontró 'procesador_nlp.py'")
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
    print("❌ ADVERTENCIA chatbot_logic: Modelo ML no encontrado.")
    modelo_noshow, preprocesador_noshow, ml_cargado = None, None, False
except Exception as e:
    print(f"❌ Error chatbot_logic al cargar modelo ML: {e}")
    modelo_noshow, preprocesador_noshow, ml_cargado = None, None, False


# --- Lógica de Predicción No-Show (Copia de app.py) ---
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
        prob = modelo_noshow.predict_proba(datos_procesados)[0][1]
        print(f"📈 chatbot_logic: Predicción No-Show ({fecha_str} {hora_str}): {prob:.2f}"); return prob
    except Exception as e: print(f"❌ chatbot_logic: Error en predicción: {e}"); return None


# --- Lógica Principal del Chatbot ---
def responder_chatbot(mensaje, historial_chat, estado_actual):
    """Función principal del chatbot con flujo conversacional para agendar."""
    print(f"\nMensaje: {mensaje}")
    if estado_actual is None: estado_actual = {}
    print(f"Estado IN: {estado_actual}")

    # Campos requeridos
    campos_paciente = ["DNI", "Nombre", "Telefono", "Email"]
    campos_cita = ["Fecha", "Hora", "Medico"]
    todos_campos = campos_paciente + campos_cita

    # 1. Procesar NLP
    if not nlp_cargado: # Verificar si el módulo NLP se importó correctamente
        return "Error: El módulo NLP no está disponible.", {}
    intencion, entidades_raw = procesar_texto(mensaje)
    print(f"Intención RAW: {intencion}, Entidades RAW: {entidades_raw}")

    # Asumir respuesta si NLP no detecta el campo esperado
    campo_preguntado_antes = estado_actual.get("campo_preguntado")
    valor_ingresado_manualmente = None
    if campo_preguntado_antes and campo_preguntado_antes.lower() not in [k.lower() for k in entidades_raw.keys()]:
        valor_ingresado = mensaje.strip()
        campo_limpio = campo_preguntado_antes
        if valor_ingresado:
             if campo_limpio == 'DNI': valor_ingresado = ''.join(filter(str.isdigit, valor_ingresado))
             if campo_limpio == 'Telefono': valor_ingresado = ''.join(filter(str.isdigit, valor_ingresado))
             if campo_limpio == 'Nombre': valor_ingresado = valor_ingresado.title()
             valor_ingresado_manualmente = valor_ingresado
             print(f"Valor para '{campo_limpio}' inferido manualmente: {valor_ingresado}")

    # Limpiar entidades NLP
    entidades_limpias = {}
    if 'dni' in entidades_raw: entidades_limpias['DNI'] = ''.join(filter(str.isdigit, str(entidades_raw.get('dni'))))
    if 'telefono' in entidades_raw: entidades_limpias['Telefono'] = ''.join(filter(str.isdigit, str(entidades_raw.get('telefono'))))
    if 'nombre' in entidades_raw: entidades_limpias['Nombre'] = entidades_raw.get('nombre', '').title()
    if 'fecha' in entidades_raw: entidades_limpias['Fecha'] = entidades_raw.get('fecha')
    if 'hora' in entidades_raw: entidades_limpias['Hora'] = entidades_raw.get('hora')
    if 'medico' in entidades_raw: entidades_limpias['Medico'] = entidades_raw.get('medico')

    # Actualizar estado (NLP + valor manual)
    estado_actual.update(entidades_limpias)
    if campo_preguntado_antes and valor_ingresado_manualmente:
        estado_actual[campo_preguntado_antes] = valor_ingresado_manualmente
    estado_actual.pop("campo_preguntado", None)
    print(f"Estado después de Update: {estado_actual}")

    # Preserve intent
    if not estado_actual.get("intent") or intencion != "desconocido":
         estado_actual["intent"] = intencion

    # --- Lógica Principal ---
    respuesta = ""
    accion_completada = False # Para saber si reiniciar estado al final

    if estado_actual.get("intent") == "agendar":
        if not flujo_cargado: # Verificar si el módulo de flujo se cargó
            return "Error: La lógica de agendamiento no está disponible.", {}

        # A. Verificar DNI y Paciente
        dni_actual = estado_actual.get("DNI"); paciente_confirmado = estado_actual.get("paciente_confirmado", False)
        id_paciente_existente = estado_actual.get("ID_Paciente"); esperando_respuesta_sino = estado_actual.get("esperando_respuesta_sino", False)

        if dni_actual and not paciente_confirmado and not id_paciente_existente and not esperando_respuesta_sino:
            paciente_encontrado = buscar_paciente_por_dni(dni_actual) # Llama a la función importada
            if paciente_encontrado:
                estado_actual["paciente_encontrado"] = paciente_encontrado; estado_actual["esperando_respuesta_sino"] = True
                respuesta = f"Encontré a {paciente_encontrado['Nombre']} con DNI {dni_actual}. ¿Eres tú? (Sí/No)"
            else: estado_actual["paciente_confirmado"] = "nuevo"
        elif esperando_respuesta_sino:
             respuesta_usuario = mensaje.lower(); paciente_guardado = estado_actual.get("paciente_encontrado", {})
             nombre_encontrado = paciente_guardado.get("Nombre", "Desconocido"); dni_conflicto = estado_actual.get("DNI")
             if "sí" in respuesta_usuario or "si" in respuesta_usuario:
                 estado_actual["ID_Paciente"] = paciente_guardado.get("ID_Paciente"); estado_actual["Nombre"] = nombre_encontrado
                 estado_actual["Telefono"] = paciente_guardado.get("Telefono"); estado_actual["Email"] = paciente_guardado.get("Email")
                 estado_actual["paciente_confirmado"] = True; estado_actual.pop("esperando_respuesta_sino", None); estado_actual.pop("paciente_encontrado", None)
                 print(f"Paciente confirmado: {estado_actual['ID_Paciente']}")
             elif "no" in respuesta_usuario:
                  respuesta = (f"Entendido. El DNI {dni_conflicto} ya está registrado a nombre de {nombre_encontrado}. Contacta a soporte.")
                  estado_actual = {"intent": "conflicto_dni"}; accion_completada = True
             else: respuesta = "¿Disculpa? ¿Confirmas que eres tú? (Sí/No)"

        # B. Recolectar Datos Faltantes
        if not respuesta and estado_actual.get("intent") != "conflicto_dni":
            campo_faltante = None
            campos_a_pedir = todos_campos if estado_actual.get("paciente_confirmado") != True else campos_cita
            for campo in campos_a_pedir:
                if not estado_actual.get(campo): campo_faltante = campo; break

            if campo_faltante:
                estado_actual["campo_preguntado"] = campo_faltante # Guardar qué preguntamos
                if campo_faltante == "DNI": respuesta = "¿Cuál es tu número de DNI?"
                elif campo_faltante == "Nombre": respuesta = "¿Cuál es tu nombre completo?"
                elif campo_faltante == "Telefono": respuesta = "¿Cuál es tu número de teléfono?"
                elif campo_faltante == "Email": respuesta = "¿Cuál es tu correo electrónico?"
                elif campo_faltante == "Fecha": respuesta = "¿Para qué fecha? (AAAA-MM-DD)"
                elif campo_faltante == "Hora": respuesta = "¿A qué hora? (HH:MM)"
                elif campo_faltante == "Medico":
                    if flujo_cargado: medicos = obtener_medicos() # Llama a la función importada
                    else: medicos = ["Error"]
                    respuesta = f"¿Con qué médico? ({', '.join(medicos)})"
                else: respuesta = f"Necesito tu {campo_faltante}."
            else: # ¡Tenemos todos los datos!
                # C. Mostrar Resumen y Pedir Confirmación
                if not estado_actual.get("esperando_confirmacion_final", False):
                    estado_actual["esperando_confirmacion_final"] = True
                    respuesta = "Revisa los datos:\n```\n" + \
                                f"Paciente: {estado_actual.get('Nombre')} | DNI: {estado_actual.get('DNI')} | Tel: {estado_actual.get('Telefono')} | Email: {estado_actual.get('Email')}\n" + \
                                f"Cita: {estado_actual.get('Fecha')} a las {estado_actual.get('Hora')} con {estado_actual.get('Medico')}\n```\n" + \
                                "¿Agendamos? (Sí/No)"
                else:
                    respuesta_usuario = mensaje.lower()
                    if "sí" in respuesta_usuario or "si" in respuesta_usuario:
                        if flujo_cargado: # Verificar si se puede agendar
                            resultado = agendar(nombre=estado_actual.get('Nombre'), dni=estado_actual.get('DNI'), telefono=estado_actual.get('Telefono'), email=estado_actual.get('Email'),
                                                fecha=estado_actual.get('Fecha'), hora=estado_actual.get('Hora'), medico=estado_actual.get('Medico'))
                            respuesta = resultado
                            if resultado and "¡Éxito!" in resultado: # Añadir predicción
                                prob_noshow = predecir_noshow(estado_actual.get('Fecha'), estado_actual.get('Hora')) # Llama a la función local
                                if prob_noshow is not None:
                                    if prob_noshow > 0.6: respuesta += f"\n⚠️ **Advertencia:** Riesgo de ausencia: {prob_noshow:.0%}"
                                    else: respuesta += f"\n(Riesgo bajo: {prob_noshow:.0%})"
                        else:
                             respuesta = "Error: La función de agendamiento no está disponible."
                        accion_completada = True
                    elif "no" in respuesta_usuario: respuesta = "Agendamiento cancelado."; accion_completada = True
                    else: respuesta = "¿Confirmamos (Sí) o cancelamos (No)?"

    # --- Lógica para otras intenciones ---
    elif estado_actual.get("intent") == "consultar":
        if not flujo_cargado: return "Error: Lógica de consulta no disponible.", {}
        dni = estado_actual.get("DNI") or entidades_limpias.get("DNI")
        if not dni: respuesta = "Necesito tu DNI para consultar."
        else:
            res_crud = consultar_citas(dni) # Llama a la función importada
            if isinstance(res_crud, list):
                if not res_crud: respuesta = f"No encontré citas para DNI {dni}."
                else:
                    respuesta = f"He encontrado {len(res_crud)} citas para DNI {dni}:\n";
                    for c in res_crud: respuesta += f"- {c.get('ID_Cita','N/A')} el {c.get('Fecha','N/A')} {c.get('Hora','N/A')} ({c.get('Estado','N/A')})\n"
            else: respuesta = str(res_crud)
        accion_completada = True

    elif estado_actual.get("intent") == "cancelar":
        if not flujo_cargado: return "Error: Lógica de cancelación no disponible.", {}
        dni = estado_actual.get("DNI") or entidades_limpias.get("DNI")
        fecha = estado_actual.get("Fecha") or entidades_limpias.get("Fecha")
        if not dni or not fecha: respuesta = "Necesito DNI y fecha para cancelar."
        else: respuesta = cancelar_cita(dni, fecha) # Llama a la función importada
        accion_completada = True

    elif estado_actual.get("intent") == "desconocido":
        respuesta = "No entendí. Intenta: agendar, consultar o cancelar."; accion_completada = True

    elif not respuesta: # Si no hizo match con nada y no se generó respuesta
         respuesta = "Hola. ¿Cómo puedo ayudarte?"; accion_completada = True

    # Reiniciar estado si la acción se completó
    if accion_completada: estado_actual = {}

    print(f"Estado OUT: {estado_actual}")
    print(f"Respuesta Final: {respuesta}")
    return respuesta, estado_actual
