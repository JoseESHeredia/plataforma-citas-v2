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
from TTS.api import TTS 

# ⭐️ NUEVAS IMPORTACIONES PARA S4-02 ⭐️
import qrcode
import io
# ------------------------------------

# ============================================================
# ⭐️ CAMBIO S4-01: Inicializar el modelo TTS (Coqui)
# ============================================================
try:
    print("Cargando modelo TTS (Coqui)...")
    tts_model = TTS(model_name="tts_models/es/css10/vits", progress_bar=True, gpu=False)
    tts_cargado = True
    print("✅ Modelo TTS cargado.")
except Exception as e:
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
# ⭐️ NUEVA LÓGICA S4-02: Generación de QR de WhatsApp
# ============================================================
def generar_qr_whatsapp(dni, fecha, hora):
    """
    Genera un código QR que enlaza a un chat de WhatsApp con un mensaje prellenado.
    Devuelve la ruta al archivo temporal de la imagen QR.
    """
    if not dni or not fecha or not hora:
        return None, "Error: DNI, Fecha y Hora son requeridos para generar el QR."

    # 📞 Número de ejemplo para el asistente (debe ser un número real con código de país)
    NUMERO_ASISTENTE = "51999888777" # Ejemplo Perú +51999888777

    # Mensaje prellenado para la confirmación
    mensaje = f"Hola, confirmo mi cita para el DNI {dni} el día {fecha} a las {hora}. ¡Gracias!"
    
    # 🔗 Generar URL de WhatsApp
    wa_url = f"https://wa.me/{NUMERO_ASISTENTE}?text={mensaje.replace(' ', '%20').replace('¡', '')}"

    try:
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=10,
            border=4,
        )
        qr.add_data(wa_url)
        qr.make(fit=True)

        img = qr.make_image(fill_color="black", back_color="white")
        
        # Guardar la imagen en un archivo temporal para que Gradio pueda mostrarla
        temp_qr_path = os.path.join(tempfile.gettempdir(), f"qr_{dni}_{fecha}.png")
        img.save(temp_qr_path)
        
        return temp_qr_path, f"QR generado con éxito. Escanea para confirmar la cita: {fecha} a las {hora}."

    except Exception as e:
        return None, f"❌ Error al generar el QR: {e}"


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


def generar_audio_respuesta(texto_respuesta):
    """Genera un archivo WAV a partir del texto usando TTS."""
    if not tts_cargado or not texto_respuesta or texto_respuesta.startswith("❌"):
        return None 
    try:
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
            audio_gen = None 
            
            if not mensaje:
                return historial, estado, gr.update(value=""), None
            
            if not chatbot_cargado:
                respuesta = "❌ Chatbot no cargado."
                audio_gen = None
            else:
                respuesta, nuevo_estado = responder_chatbot(mensaje, historial, estado)
                audio_gen = generar_audio_respuesta(respuesta) 

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
        
        btn_enviar_texto.click(fn=manejar_texto, inputs=[entrada_texto, chatbot, estado_conversacion],
                               outputs=[chatbot, estado_conversacion, entrada_texto, audio_respuesta])
        entrada_texto.submit(fn=manejar_texto, inputs=[entrada_texto, chatbot, estado_conversacion],
                             outputs=[chatbot, estado_conversacion, entrada_texto, audio_respuesta])
   
    # --------------------------------------------------------
    # 📱 NUEVA PESTAÑA S4-02: QR de Confirmación
    # --------------------------------------------------------
    with gr.Tab("QR de Confirmación"):
        gr.Markdown("### 📱 Generar QR para confirmar cita por WhatsApp")
        gr.Markdown("Ingresa los datos de la cita agendada para generar el código.")

        with gr.Row():
            txt_dni_qr = gr.Textbox(label="DNI del Paciente", placeholder="Ej: 12345678")
            txt_fecha_qr = gr.Textbox(label="Fecha de la Cita", placeholder="AAAA-MM-DD")
            txt_hora_qr = gr.Textbox(label="Hora de la Cita", placeholder="HH:MM")
        
        btn_generar_qr = gr.Button("Generar QR", variant="primary")
        
        qr_output_img = gr.Image(label="Código QR (Escanea para confirmar)", type="filepath")
        qr_output_msg = gr.Label(label="Mensaje de Estado")

        btn_generar_qr.click(
            fn=generar_qr_whatsapp,
            inputs=[txt_dni_qr, txt_fecha_qr, txt_hora_qr],
            outputs=[qr_output_img, qr_output_msg]
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
    demo.queue().launch(server_name="0.0.0.0", server_port=7860)
