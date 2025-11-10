# ============================================================
# 🩺 Plataforma de Citas Médicas v2
# ============================================================

import gradio as gr
import pandas as pd
import numpy as np
import joblib
import gspread
import tempfile
import scipy.io.wavfile as wavfile
from datetime import date
import os 
import re 
# from TTS.api import TTS # ⭐️ CAMBIO S4-01: Importación movida al bloque try/except

# ============================================================
# ⭐️ CAMBIO S4-01: Inicializar el modelo TTS (Coqui)
# ============================================================
# Esto puede tardar la primera vez que se ejecuta mientras descarga el modelo.
# ⭐️ INICIO DE LA CORRECCIÓN: Se envuelve la importación Y la inicialización
# Esto captura el 'OSError' local de Windows sin detener la app.
try:
    from TTS.api import TTS # ⭐️ CAMBIO S4-01: Importación movida aquí
    print("Cargando modelo TTS (Coqui)...")
    # Este es un modelo en español rápido y de calidad decente:
    tts_model = TTS(model_name="tts_models/es/css10/vits", progress_bar=True, gpu=False)
    tts_cargado = True
    print("✅ Modelo TTS cargado.")
except (OSError, Exception) as e: # Captura el OSError de Windows y otros errores
    # Si falla (ej. en una máquina sin internet), la app seguirá funcionando sin voz.
    print(f"❌ ADVERTENCIA: No se pudo cargar el modelo TTS: {e}")
    tts_model = None
    tts_cargado = False
# ⭐️ FIN DE LA CORRECCIÓN

# ============================================================
# 🔧 Importaciones de Lógica
# ============================================================

# --- flujo_agendamiento ---
try:
    from flujo_agendamiento import (
        agendar,
        consultar_citas,
        cancelar_cita,
        obtener_medicos,
        pacientes_sheet,
        citas_sheet,
        buscar_paciente_por_dni
    )
    flujo_cargado = True
    print("✅ Módulos CRUD y búsqueda cargados.")
except ImportError as e:
    print(f"❌ ERROR FATAL: No se pudo importar 'flujo_agendamiento.py': {e}")
    flujo_cargado = False

    def agendar(*args): return "Error importación flujo_agendamiento"
    def consultar_citas(dni): return "Error importación flujo_agendamiento"
    def cancelar_cita(dni, fecha): return "Error importación flujo_agendamiento"
    def obtener_medicos(): return ["Error"]
    def buscar_paciente_por_dni(dni): return None
    pacientes_sheet = None
    citas_sheet = None


# --- chatbot_logic ---
try:
    # Usamos la última versión corregida de la lógica
    from chatbot_logic import responder_chatbot, predecir_noshow
    chatbot_cargado = True
    print("✅ Módulo 'chatbot_logic.py' cargado.")
except ImportError as e:
    print(f"❌ ERROR FATAL: No se pudo importar 'chatbot_logic.py': {e}")
    chatbot_cargado = False

    def responder_chatbot(m, h, s): return f"Error importación chatbot_logic: {e}", {}
    def predecir_noshow(f, h): return None


# --- transcriptor ---
try:
    from transcriptor import transcribir_audio
    stt_cargado = True
    print("✅ Módulo 'transcriptor.py' cargado.")
except ImportError:
    print("⚠️ ADVERTENCIA: 'transcriptor.py' no encontrado. Usando placeholder.")
    stt_cargado = False

    def transcribir_audio_placeholder(audio): return "[Transcripción no disponible]"
    transcribir_audio = transcribir_audio_placeholder


# ============================================================
# 📊 Lógica de Carga de Datos (Google Sheets)
# ============================================================

def cargar_datos_gsheets():
    """Carga datos de GSheets para mostrar en tabla."""
    print("🔄 app.py: Cargando datos desde GSheets para tabla...")

    default_cols_pacientes = ["ID_Paciente", "Nombre", "DNI", "Telefono", "Email"]
    default_cols_citas = ["ID_Cita", "ID_Paciente", "Fecha", "Hora", "Medico", "Especialidad", "Estado"]

    df_pacientes = pd.DataFrame(columns=default_cols_pacientes)
    df_citas = pd.DataFrame(columns=default_cols_citas)

    try:
        if pacientes_sheet:
            vals = pacientes_sheet.get_all_values()
            if len(vals) > 1:
                df_pacientes = pd.DataFrame(vals[1:], columns=vals[0])
        if citas_sheet:
            vals = citas_sheet.get_all_values()
            if len(vals) > 1:
                df_citas = pd.DataFrame(vals[1:], columns=vals[0])
    except Exception as e:
        print(f"❌ app.py: Error al leer GSheets para tabla: {e}")

    print("✅ app.py: Datos cargados para tabla.")
    return df_pacientes.tail(10), df_citas.tail(10)


# ============================================================
# 🧩 Wrappers
# ============================================================

def agendar_manual_y_predecir(nombre, dni, telefono, email, fecha_str, hora_str, medico):
    """Agendar cita y mostrar predicción de no-show."""
    res = agendar(nombre, dni, telefono, email, fecha_str, hora_str, medico)
    if res and "¡Éxito!" in res:
        prob = predecir_noshow(fecha_str, hora_str)
        if prob is not None:
            res += f"\n{'⚠️ Riesgo ausencia:' if prob > 0.6 else '(Riesgo bajo:'} {prob:.0%})"
    return res


def consultar_citas_gradio(dni):
    """Consultar citas y devolver texto formateado."""
    resultado = consultar_citas(dni)
    if isinstance(resultado, list):
        if not resultado:
            return f"No hay citas para DNI {dni}."
        res_txt = f"Citas para DNI {dni} ({len(resultado)}):\n"
        for c in resultado:
            res_txt += f"- ID:{c.get('ID_Cita', 'N/A')}, {c.get('Fecha', 'N/A')} {c.get('Hora', 'N/A')} ({c.get('Estado', 'N/A')})\n"
        return res_txt
    return str(resultado)

# ⭐️ CAMBIO S4-01: Nueva función para generar el audio
def generar_audio_respuesta(texto_respuesta):
    """Genera un archivo WAV a partir del texto usando TTS."""
    if not tts_cargado or not texto_respuesta or texto_respuesta.startswith("❌"):
        return None # No generar audio si no hay TTS o si es un error
    try:
        # Usamos un archivo temporal para la respuesta
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio_file:
            tts_model.tts_to_file(text=texto_respuesta, file_path=temp_audio_file.name)
            print(f"🔊 Audio de respuesta generado en: {temp_audio_file.name}")
            return temp_audio_file.name
    except Exception as e:
        print(f"❌ Error al generar audio TTS: {e}")
        return None

# ============================================================
# 🤖 Lógica Central del Chatbot (Manejadores)
# ============================================================

# --- Manejador de Texto (Usado por Pestaña Híbrida) ---
def manejar_texto(mensaje, historial, estado):
    audio_gen = None 
    if not mensaje:
        return historial, estado, gr.update(value=""), None
    
    if not chatbot_cargado:
        respuesta = "❌ Chatbot no cargado."
    else:
        respuesta, nuevo_estado = responder_chatbot(mensaje, historial, estado)
        audio_gen = generar_audio_respuesta(respuesta) 

    return historial + [[mensaje, respuesta]], nuevo_estado, gr.update(value=""), audio_gen

# --- Manejador de Transcripción (Usado por Pestaña Híbrida) ---
def procesar_audio_a_textbox(audio_array):
    if audio_array is None or len(audio_array) == 0:
        print("❌ No se recibió audio (array vacío).")
        return gr.update(value="[No se grabó audio]")
    try:
        sample_rate, audio_data = audio_array
        duration = len(audio_data) / sample_rate
        print(f"📊 Audio recibido: sample_rate={sample_rate}, duración={duration:.2f}s, tamaño_array={len(audio_data)}")

        if duration < 1.0:
            print("⚠️ Audio demasiado corto (<1s), no se transcribe.")
            return gr.update(value="[Audio demasiado corto, graba más tiempo]")

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_path = temp_file.name
            wavfile.write(temp_path, sample_rate, audio_data.astype(np.int16))

        texto = transcribir_audio(temp_path).strip()
        if not texto:
            texto = "[No se reconoció voz]"
        print(f"📝 Transcripción obtenida: {texto}")

        os.unlink(temp_path)
        return gr.update(value=texto)

    except Exception as e:
        print(f"❌ Error en procesamiento de audio: {e}")
        return gr.update(value="[Error al procesar audio]")

# --- Manejador "Solo Voz" (Usado por Pestaña Solo Voz) ---
def manejar_solo_voz(audio_array, historial, estado):
    if audio_array is None or len(audio_array) == 0:
        print("❌ [Solo Voz] No se recibió audio.")
        return historial, estado, None
    try:
        # 1. Transcribir el audio
        sample_rate, audio_data = audio_array
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_path = temp_file.name
            wavfile.write(temp_path, sample_rate, audio_data.astype(np.int16))
        
        texto_transcrito = transcribir_audio(temp_path).strip()
        os.unlink(temp_path)
        
        if not texto_transcrito:
            texto_transcrito = "[No se reconoció voz]"

        # 2. Obtener respuesta del bot
        respuesta_bot, nuevo_estado = responder_chatbot(texto_transcrito, historial, estado)
        
        # 3. Generar audio de respuesta
        audio_gen = generar_audio_respuesta(respuesta_bot)

        # 4. Devolver todo
        return historial + [[texto_transcrito, respuesta_bot]], nuevo_estado, audio_gen
        
    except Exception as e:
        print(f"❌ Error en 'manejar_solo_voz': {e}")
        return historial + [[f"[Error: {e}]", None]], estado, None


# ============================================================
# 🧠 Interfaz Gradio
# ============================================================

with gr.Blocks(theme=gr.themes.Soft(), title="Plataforma de Citas v2") as demo:
    
    # --- Definición de Estados (uno por pestaña) ---
    estado_conversacion_hibrida = gr.State({})
    estado_conversacion_voz = gr.State({})
    
    # --- Mensaje de Bienvenida ---
    bienvenida = "¡Hola! Tu bienestar es nuestra prioridad. Para ayudarte rápido, dime si quieres **agendar** una cita, **consultar** tus horarios o **cancelar** una cita."
    historial_bienvenida = [[None, bienvenida]]
    
    # --- Textos de Botones de Acción ---
    txt_accion_agendar = "Deseo agendar una cita"
    txt_accion_consultar = "Quiero consultar mis citas"
    txt_accion_cancelar = "Quiero cancelar mi cita"


    gr.Markdown("# 🤖 Plataforma de Citas por Voz y Chat (Sprint 4)")

    # --------------------------------------------------------
    # 🗨️ PESTAÑA 1: Chatbot (Híbrido - Texto y Voz)
    # --------------------------------------------------------
    with gr.Tab("Chatbot (Híbrido)"):
        gr.Markdown("### 💬 Envía texto o audio al asistente para agendar, consultar o cancelar citas")

        # ⭐️ CAMBIO: Añadido 'value=historial_bienvenida'
        chatbot_hibrido = gr.Chatbot(
            label="Asistente Virtual",
            value=historial_bienvenida, 
            height=400, 
            bubble_full_width=False
        )
        
        audio_respuesta_hibrida = gr.Audio(label="Respuesta de Voz", autoplay=True, visible=True, type="filepath")

        with gr.Column():
            
            # ⭐️ CAMBIO: Añadidos los 3 botones de acción rápida
            with gr.Row():
                btn_accion_agendar = gr.Button(txt_accion_agendar, variant="secondary")
                btn_accion_consultar = gr.Button(txt_accion_consultar, variant="secondary")
                btn_accion_cancelar = gr.Button(txt_accion_cancelar, variant="secondary")

            entrada_texto_hibrida = gr.Textbox(
                placeholder="Escribe tu mensaje...",
                scale=7,
                lines=1,
                container=True,
                show_label=False
            )

            audio_input_hibrido = gr.Audio(
                sources=["microphone"],
                type="numpy",
                label=None,
                show_label=False,
                interactive=True,
                streaming=False,
                elem_id="mic_input_hibrido",
                autoplay=False
            )

            with gr.Row():
                btn_procesar_audio = gr.Button("Procesar Audio", variant="secondary", scale=1)
                btn_enviar_texto = gr.Button("Enviar", variant="primary", scale=1)
        
        # --- Conexiones de Pestaña Híbrida ---
        btn_procesar_audio.click(fn=procesar_audio_a_textbox, inputs=[audio_input_hibrido], outputs=[entrada_texto_hibrida])
        
        btn_enviar_texto.click(fn=manejar_texto, inputs=[entrada_texto_hibrida, chatbot_hibrido, estado_conversacion_hibrida],
                               outputs=[chatbot_hibrido, estado_conversacion_hibrida, entrada_texto_hibrida, audio_respuesta_hibrida])
        
        entrada_texto_hibrida.submit(fn=manejar_texto, inputs=[entrada_texto_hibrida, chatbot_hibrido, estado_conversacion_hibrida],
                                     outputs=[chatbot_hibrido, estado_conversacion_hibrida, entrada_texto_hibrida, audio_respuesta_hibrida])

        # ⭐️ CAMBIO: Conexiones para los 3 botones de acción rápida
        btn_accion_agendar.click(fn=manejar_texto, inputs=[gr.State(txt_accion_agendar), chatbot_hibrido, estado_conversacion_hibrida],
                                 outputs=[chatbot_hibrido, estado_conversacion_hibrida, entrada_texto_hibrida, audio_respuesta_hibrida])
        
        btn_accion_consultar.click(fn=manejar_texto, inputs=[gr.State(txt_accion_consultar), chatbot_hibrido, estado_conversacion_hibrida],
                                   outputs=[chatbot_hibrido, estado_conversacion_hibrida, entrada_texto_hibrida, audio_respuesta_hibrida])
        
        btn_accion_cancelar.click(fn=manejar_texto, inputs=[gr.State(txt_accion_cancelar), chatbot_hibrido, estado_conversacion_hibrida],
                                  outputs=[chatbot_hibrido, estado_conversacion_hibrida, entrada_texto_hibrida, audio_respuesta_hibrida])

    # --------------------------------------------------------
    # 🎙️ PESTAÑA 2: Solo Voz (Nueva)
    # --------------------------------------------------------
    with gr.Tab("Solo Voz"):
        gr.Markdown("### 🎤 Presiona Grabar, habla y suelta. El asistente te responderá con voz.")
        
        # ⭐️ CAMBIO: Añadido 'value=historial_bienvenida'
        chatbot_voz = gr.Chatbot(
            label="Asistente Virtual (Voz)", 
            value=historial_bienvenida,
            height=400, 
            bubble_full_width=False
        )
        
        audio_respuesta_voz = gr.Audio(label="Respuesta de Voz", autoplay=True, visible=True, type="filepath")
        
        audio_input_voz = gr.Audio(
            sources=["microphone"],
            type="numpy",
            label="Presiona para grabar y luego 'Stop' para enviar",
            show_label=True,
            interactive=True,
            streaming=False,
            elem_id="mic_input_voz",
            autoplay=False
        )
        
        # --- Conexión de Pestaña Solo Voz ---
        # ⭐️ CAMBIO: Se activa con 'stop_recording' para el flujo automático
        audio_input_voz.stop_recording(
            fn=manejar_solo_voz,
            inputs=[audio_input_voz, chatbot_voz, estado_conversacion_voz],
            outputs=[chatbot_voz, estado_conversacion_voz, audio_respuesta_voz]
        )


    # --------------------------------------------------------
    # 📋 PESTAÑA 3: Datos (Google Sheets)
    # --------------------------------------------------------
    with gr.Tab("Datos (Google Sheets - EN VIVO)"):
        gr.Markdown("### Últimos Registros en Google Sheets")
        gr.Markdown("⚠️ Lee de GSheets (puede tardar).")

        with gr.Row():
            df_pacientes_display = gr.DataFrame(label="Pacientes (Google Sheet)")
            df_citas_display = gr.DataFrame(label="Citas (Google Sheet)")

        btn_actualizar_datos = gr.Button("Actualizar Tablas (desde Google Sheets)")
        btn_actualizar_datos.click(fn=cargar_datos_gsheets, inputs=None,
                                   outputs=[df_pacientes_display, df_citas_display])

    # --------------------------------------------------------
    # 🧪 PESTAÑA 4: Testeo (CRUD)
    # --------------------------------------------------------
    with gr.Tab("Testeo (CRUD GSheets)"):
        gr.Markdown("### Testeo Directo de Funciones CRUD")

        with gr.Tabs():
            # --- Agendar ---
            with gr.TabItem("Agendar (Create)"):
                with gr.Row():
                    with gr.Column():
                        txt_nombre_test = gr.Textbox(label="Nombre")
                        txt_dni_test = gr.Textbox(label="DNI")
                        txt_telefono_test = gr.Textbox(label="Teléfono")
                        txt_email_test = gr.Textbox(label="Email")
                    with gr.Column():
                        fecha_default_test = date.today().strftime("%Y-%m-%d")
                        txt_fecha_test = gr.Textbox(label="Fecha", value=fecha_default_test)
                        txt_hora_test = gr.Textbox(label="Hora", value="17:00")
                        lista_medicos_test = obtener_medicos()
                        dd_medico_test = gr.Dropdown(
                            label="Médico",
                            choices=lista_medicos_test,
                            value=lista_medicos_test[0] if lista_medicos_test else None
                        )
                        btn_agendar_test = gr.Button("Agendar y Predecir", variant="primary")

                lbl_resultado_agendar = gr.Label(label="Resultado Agendar")
                btn_agendar_test.click(
                    fn=agendar_manual_y_predecir,
                    inputs=[txt_nombre_test, txt_dni_test, txt_telefono_test, txt_email_test,
                            txt_fecha_test, txt_hora_test, dd_medico_test],
                    outputs=[lbl_resultado_agendar]
                )

            # --- Consultar ---
            with gr.TabItem("Consultar (Read)"):
                txt_dni_consultar = gr.Textbox(label="DNI a Consultar")
                btn_consultar_test = gr.Button("Consultar Citas", variant="secondary")
                txt_resultado_consultar = gr.Textbox(label="Resultado Consulta", lines=5, interactive=False)
                btn_consultar_test.click(fn=consultar_citas_gradio,
                                         inputs=[txt_dni_consultar],
                                         outputs=[txt_resultado_consultar])

            # --- Cancelar ---
            with gr.TabItem("Cancelar (Update)"):
                txt_dni_cancelar = gr.Textbox(label="DNI")
                txt_fecha_cancelar = gr.Textbox(label="Fecha a Cancelar", placeholder="AAAA-MM-DD")
                btn_cancelar_test = gr.Button("Cancelar Cita", variant="stop")
                lbl_resultado_cancelar = gr.Label(label="Resultado Cancelación")
                btn_cancelar_test.click(fn=cancelar_cita,
                                        inputs=[txt_dni_cancelar, txt_fecha_cancelar],
                                        outputs=[lbl_resultado_cancelar])

    # Carga inicial de datos
    demo.load(fn=cargar_datos_gsheets, inputs=None,
              outputs=[df_pacientes_display, df_citas_display])


# ============================================================
# 🚀 Ejecución
# ============================================================

if __name__ == "__main__":
    # ⭐️ CAMBIO CRÍTICO: Usando tu código de despliegue de HF
    demo.queue().launch(server_name="0.0.0.0", server_port=7860)
