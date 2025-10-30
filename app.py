import gradio as gr
import pandas as pd
from datetime import date
import gspread # Para manejo de errores
import joblib # Para cargar el modelo ML
import numpy as np # Para manejar datos del modelo

# --- Importaciones de Lógica ---
try:
    # Importar TODO lo necesario de flujo_agendamiento
    from flujo_agendamiento import agendar, consultar_citas, cancelar_cita, obtener_medicos, pacientes_sheet, citas_sheet, buscar_paciente_por_dni
    flujo_cargado = True
    print("✅ Módulos CRUD y búsqueda cargados.")
# --- BLOQUE CORREGIDO DEFINITIVAMENTE ---
except ImportError as e:
    print(f"❌ ERROR FATAL: No se pudo importar 'flujo_agendamiento.py': {e}")
    # Define placeholders en líneas separadas
    flujo_cargado = False
    def agendar(*args): return "Error importación flujo_agendamiento"
    def consultar_citas(dni): return "Error importación flujo_agendamiento"
    def cancelar_cita(dni, fecha): return "Error importación flujo_agendamiento"
    def obtener_medicos(): return ["Error"]
    def buscar_paciente_por_dni(dni): return None
    # Asignación en línea separada
    pacientes_sheet = None
    citas_sheet = None
# --- FIN BLOQUE CORREGIDO ---

try:
    # Importar lógica del chatbot (separada)
    from chatbot_logic import responder_chatbot, predecir_noshow # También importamos predecir_noshow
    chatbot_cargado = True
    print("✅ Módulo 'chatbot_logic.py' cargado.")
except ImportError as e:
    print(f"❌ ERROR FATAL: No se pudo importar 'chatbot_logic.py': {e}")
    chatbot_cargado = False
    # Placeholders en líneas separadas
    def responder_chatbot(m, h, s): return f"Error importación chatbot_logic: {e}", {}
    def predecir_noshow(f, h): return None

try:
    # Importar transcriptor
    from transcriptor import transcribir_audio
    stt_cargado = True
    print("✅ Módulo 'transcriptor.py' cargado.")
except ImportError:
    print("ADVERTENCIA: 'transcriptor.py' no encontrado. Usando placeholder.")
    stt_cargado = False
    # Placeholder en línea separada
    def transcribir_audio_placeholder(audio): return "[Transcripción no disponible]"
    transcribir_audio = transcribir_audio_placeholder


# --- Lógica de Carga de Datos (GSheets en vivo) ---
def cargar_datos_gsheets():
    """Carga datos de GSheets para mostrar en tabla."""
    print("🔄 app.py: Cargando datos desde GSheets para tabla...")
    default_cols_pacientes=["ID_Paciente","Nombre","DNI","Telefono","Email"]
    default_cols_citas=["ID_Cita","ID_Paciente","Fecha","Hora","Medico","Especialidad","Estado"]
    df_pacientes=pd.DataFrame(columns=default_cols_pacientes); df_citas=pd.DataFrame(columns=default_cols_citas)
    try:
        if pacientes_sheet: vals=pacientes_sheet.get_all_values();
        if len(vals)>1: df_pacientes=pd.DataFrame(vals[1:],columns=vals[0])
        if citas_sheet: vals=citas_sheet.get_all_values();
        if len(vals)>1: df_citas=pd.DataFrame(vals[1:],columns=vals[0])
    except Exception as e: print(f"❌ app.py: Error al leer GSheets para tabla: {e}")
    print("✅ app.py: Datos cargados para tabla."); return df_pacientes.tail(10), df_citas.tail(10)

# --- Wrapper para Agendar Manual + Predicción ---
def agendar_manual_y_predecir(nombre, dni, telefono, email, fecha_str, hora_str, medico):
    res = agendar(nombre, dni, telefono, email, fecha_str, hora_str, medico)
    if res and "¡Éxito!" in res: prob = predecir_noshow(fecha_str, hora_str);
    if prob is not None: res += f"\n{'⚠️ Riesgo ausencia:' if prob>0.6 else '(Riesgo bajo:'} {prob:.0%})"
    return res

# --- Wrapper para Consultar Citas (Formateo) ---
def consultar_citas_gradio(dni):
    resultado = consultar_citas(dni);
    if isinstance(resultado, list):
        if not resultado: return f"No citas para DNI {dni}."
        res_txt = f"Citas para DNI {dni} ({len(resultado)}):\n";
        for c in resultado: res_txt += f"- ID:{c.get('ID_Cita','N/A')}, {c.get('Fecha','N/A')} {c.get('Hora','N/A')} ({c.get('Estado','N/A')})\n"
        return res_txt
    else: return str(resultado)

# --- Wrapper para Transcribir y Responder ---
def transcribir_y_responder(audio_path, historial_chat_actual, estado_actual):
    """Primero transcribe audio, luego llama al chatbot."""
    print(f"🎙️ Recibido audio: {audio_path}")
    if audio_path is None: return "[No audio]", "[Esperando]", estado_actual or {}
    texto_transcrito = transcribir_audio(audio_path); print(f"📝 Texto: {texto_transcrito}")
    if texto_transcrito.startswith("❌") or texto_transcrito.startswith("⚠️"): return texto_transcrito, f"Error: {texto_transcrito}", estado_actual or {}
    if chatbot_cargado: resp_bot, n_estado = responder_chatbot(texto_transcrito, historial_chat_actual, estado_actual)
    else: resp_bot, n_estado = "Error: Chatbot no cargado.", estado_actual or {}
    return texto_transcrito, resp_bot, n_estado

# --- Construcción de la Interfaz ---
with gr.Blocks(theme=gr.themes.Soft(), title="Plataforma de Citas v2") as demo:
    estado_conversacion = gr.State({}) # Estado compartido
    gr.Markdown("# 🤖 Plataforma de Citas por Voz y Chat (Sprint 3)")

    # --- PESTAÑA 1: CHATBOT ---
    with gr.Tab("Chatbot (NLP)"):
        gr.Markdown("### Conversa para agendar, consultar o cancelar")
        gr.ChatInterface(fn=responder_chatbot if chatbot_cargado else None, chatbot=gr.Chatbot(height=400),
                         textbox=gr.Textbox(placeholder="Escribe tu solicitud aquí...", container=False, scale=7),
                         title="Asistente Virtual de Citas",
                         examples=[["Agendar cita Dr.Perez mañana", {}], ["Ver mis citas dni 98765432", {}], ["cancelar cita 98765432 para 2025-10-30", {}]],
                         additional_inputs=[estado_conversacion], additional_outputs=[estado_conversacion])

    # --- PESTAÑA 2: VOZ ---
    with gr.Tab("Voz (STT + Chatbot)"):
        gr.Markdown("### Habla con el Asistente Virtual")
        with gr.Row(): audio_input = gr.Audio(sources=["microphone"], type="filepath", label="Graba tu solicitud", scale=2); btn_procesar_voz = gr.Button("Procesar Voz", variant="primary")
        gr.Markdown("---")
        with gr.Row(): txt_transcripcion = gr.Textbox(label="Texto Transcrito", interactive=False); txt_respuesta_bot = gr.Textbox(label="Respuesta del Asistente", interactive=False)
        btn_procesar_voz.click(fn=transcribir_y_responder, inputs=[audio_input, gr.State(None), estado_conversacion], outputs=[txt_transcripcion, txt_respuesta_bot, estado_conversacion])

    # --- PESTAÑA 3: DATOS ---
    with gr.Tab("Datos (Google Sheets - EN VIVO)"):
        gr.Markdown("### Últimos Registros en Google Sheets")
        gr.Markdown("⚠️ Lee de GSheets (puede tardar).")
        with gr.Row(): # Asegurar indentación correcta
            df_pacientes_display = gr.DataFrame(label="Pacientes (Google Sheet)")
            df_citas_display = gr.DataFrame(label="Citas (Google Sheet)")
        btn_actualizar_datos = gr.Button("Actualizar Tablas (desde Google Sheets)")
        btn_actualizar_datos.click(fn=cargar_datos_gsheets, inputs=None, outputs=[df_pacientes_display, df_citas_display]) # Corchete cerrado

    # --- PESTAÑA 4: TESTEO (CRUD) ---
    with gr.Tab("Testeo (CRUD GSheets)"):
        gr.Markdown("### Testeo Directo de Funciones CRUD")
        with gr.Tabs():
             with gr.TabItem("Agendar (Create)"):
                 with gr.Row():
                      with gr.Column(): txt_nombre_test=gr.Textbox(label="Nombre"); txt_dni_test=gr.Textbox(label="DNI"); txt_telefono_test=gr.Textbox(label="Teléfono"); txt_email_test=gr.Textbox(label="Email")
                      with gr.Column(): fecha_default_test=date.today().strftime("%Y-%m-%d"); txt_fecha_test=gr.Textbox(label="Fecha", value=fecha_default_test); txt_hora_test=gr.Textbox(label="Hora", value="17:00"); lista_medicos_test=obtener_medicos(); dd_medico_test=gr.Dropdown(label="Médico", choices=lista_medicos_test, value=lista_medicos_test[0] if lista_medicos_test else None); btn_agendar_test=gr.Button("Agendar y Predecir", variant="primary")
                 lbl_resultado_agendar=gr.Label(label="Resultado Agendar"); btn_agendar_test.click(fn=agendar_manual_y_predecir, inputs=[txt_nombre_test, txt_dni_test, txt_telefono_test, txt_email_test, txt_fecha_test, txt_hora_test, dd_medico_test], outputs=[lbl_resultado_agendar])
             with gr.TabItem("Consultar (Read)"):
                 txt_dni_consultar=gr.Textbox(label="DNI a Consultar"); btn_consultar_test=gr.Button("Consultar Citas", variant="secondary"); txt_resultado_consultar=gr.Textbox(label="Resultado Consulta", lines=5, interactive=False); btn_consultar_test.click(fn=consultar_citas_gradio, inputs=[txt_dni_consultar], outputs=[txt_resultado_consultar])
             with gr.TabItem("Cancelar (Update)"):
                 txt_dni_cancelar=gr.Textbox(label="DNI"); txt_fecha_cancelar=gr.Textbox(label="Fecha a Cancelar", placeholder="AAAA-MM-DD"); btn_cancelar_test=gr.Button("Cancelar Cita", variant="stop"); lbl_resultado_cancelar=gr.Label(label="Resultado Cancelación"); btn_cancelar_test.click(fn=cancelar_cita, inputs=[txt_dni_cancelar, txt_fecha_cancelar], outputs=[lbl_resultado_cancelar])

    # Carga inicial de datos
    demo.load(fn=cargar_datos_gsheets, inputs=None, outputs=[df_pacientes_display, df_citas_display])

# --- Lanzar la aplicación ---
if __name__ == "__main__":
    print("Intentando lanzar la aplicación Gradio...")
    try:
        demo.queue().launch(server_name="127.0.0.1", server_port=7860)
        print("¡Aplicación lanzada! Accede en http://127.0.0.1:7860")
    except Exception as e:
        print(f"❌ ERROR al lanzar Gradio: {e}")
