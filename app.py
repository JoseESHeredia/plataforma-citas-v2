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
from TTS.api import TTS # ⭐️ CAMBIO S4-01: Nueva importación

# ============================================================
# ⭐️ CAMBIO S4-01: Inicializar el modelo TTS (Coqui)
# ============================================================
# Esto puede tardar la primera vez que se ejecuta mientras descarga el modelo.
try:
    print("Cargando modelo TTS (Coqui)...")
    # Este es un modelo en español rápido y de calidad decente:
    tts_model = TTS(model_name="tts_models/es/css10/vits", progress_bar=True, gpu=False)
    tts_cargado = True
    print("✅ Modelo TTS cargado.")
except Exception as e:
    # Si falla (ej. en una máquina sin internet), la app seguirá funcionando sin voz.
    print(f"❌ ADVERTENCIA: No se pudo cargar el modelo TTS: {e}")
    tts_model = None
    tts_cargado = False

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


def transcribir_y_responder(audio_path, historial_chat_actual, estado_actual):
    # Esta función es la antigua de "Voz". La mantenemos por si se usa en otro lado.
    # La nueva lógica de audio está en "procesar_audio_a_textbox"
    print(f"🎙️ Recibido audio: {audio_path}")
    if audio_path is None:
        return "[No audio]", "[Esperando]", estado_actual or {}

    texto_transcrito = transcribir_audio(audio_path)
    print(f"📝 Texto: {texto_transcrito}")

    if texto_transcrito.startswith("❌") or texto_transcrito.startswith("⚠️"):
        return texto_transcrito, f"Error: {texto_transcrito}", estado_actual or {}

    if chatbot_cargado:
        resp_bot, n_estado = responder_chatbot(texto_transcrito, historial_chat_actual, estado_actual)
    else:
        resp_bot, n_estado = "Error: Chatbot no cargado.", estado_actual or {}

    return texto_transcrito, resp_bot, n_estado

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
# 🧠 Interfaz Gradio
# ============================================================

with gr.Blocks(theme=gr.themes.Soft(), title="Plataforma de Citas v2") as demo:
    estado_conversacion = gr.State({})

    gr.Markdown("# 🤖 Plataforma de Citas por Voz y Chat (Sprint 4)")

    # --------------------------------------------------------
    # 🗨️ PESTAÑA 1: Chatbot (Texto + Audio)
    # --------------------------------------------------------
    with gr.Tab("Chatbot (NLP)"):
        gr.Markdown("### 💬 Envía texto o audio al asistente para agendar, consultar o cancelar citas")

        chatbot = gr.Chatbot(label="Asistente Virtual", height=400, bubble_full_width=False)
        
        # ⭐️ CAMBIO S4-01: Componente de audio para la respuesta
        audio_respuesta = gr.Audio(label="Respuesta de Voz", autoplay=True, visible=True, type="filepath")

        with gr.Column():
            entrada_texto = gr.Textbox(
                placeholder="Escribe tu mensaje...",
                scale=7,
                lines=1,
                container=True,
                show_label=False
            )

            audio_input = gr.Audio(
                sources=["microphone"],
                type="numpy",
                label=None,
                show_label=False,
                interactive=True,
                streaming=False,
                elem_id="mic_input",
                autoplay=False
            )

            with gr.Row():
                btn_procesar_audio = gr.Button("Procesar Audio", variant="secondary", scale=1)
                btn_enviar_texto = gr.Button("Enviar", variant="primary", scale=1)

        # --- Funciones internas ---
        def manejar_texto(mensaje, historial, estado):
            # Limpiar el audio anterior
            audio_gen = None 
            
            if not mensaje:
                return historial, estado, gr.update(value=""), None
            
            if not chatbot_cargado:
                respuesta = "❌ Chatbot no cargado."
                audio_gen = None
            else:
                respuesta, nuevo_estado = responder_chatbot(mensaje, historial, estado)
                # ⭐️ CAMBIO S4-01: Generar audio de la respuesta
                audio_gen = generar_audio_respuesta(respuesta) 

            # ⭐️ CAMBIO S4-01: Devolver el audio generado
            return historial + [[mensaje, respuesta]], nuevo_estado, gr.update(value=""), audio_gen

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

        # --- Conexiones ---
        btn_procesar_audio.click(fn=procesar_audio_a_textbox, inputs=[audio_input], outputs=[entrada_texto])
        
        # ⭐️ CAMBIO S4-01: Añadir 'audio_respuesta' a las salidas
        btn_enviar_texto.click(fn=manejar_texto, inputs=[entrada_texto, chatbot, estado_conversacion],
                               outputs=[chatbot, estado_conversacion, entrada_texto, audio_respuesta])
        entrada_texto.submit(fn=manejar_texto, inputs=[entrada_texto, chatbot, estado_conversacion],
                             outputs=[chatbot, estado_conversacion, entrada_texto, audio_respuesta])
   

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
